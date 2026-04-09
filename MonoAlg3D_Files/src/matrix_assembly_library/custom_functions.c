#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../3dparty/sds/sds.h"
#include "../3dparty/stb_ds.h"
#include "../alg/grid/grid.h"
#include "../config/assembly_matrix_config.h"
#include "../libraries_common/common_data_structures.h"
#include "../utils/file_utils.h"
#include "../utils/utils.h"
#include "../domains_library/custom_mesh_info_data.h"

#include "assembly_common.c"

#ifdef COMPILE_CUDA
#include "../gpu_utils/gpu_utils.h"
#endif

ASSEMBLY_MATRIX(anisotropic_sigma_assembly_matrix_for_hu_mesh) {

    if(the_grid->adaptive) {
        log_error_and_exit("anisotropic_sigma_assembly_matrix function does not support mesh adaptivity yet!. Aborting!\n");
    }

    uint32_t num_active_cells = the_grid->num_active_cells;
    struct cell_node **ac = the_grid->active_cells;

    initialize_diagonal_elements(the_solver, the_grid);

    //      D tensor    //
    // | sx    sxy   sxz |
    // | sxy   sy    syz |
    // | sxz   syz   sz  |
    real_cpu D[3][3];
    int i;

    real_cpu sigma_l = 0.0;
    real_cpu sigma_t = 0.0;
    real_cpu sigma_n = 0.0;
    real_cpu sigma_C = 1.0;
    
    char *fiber_file = NULL;
    GET_PARAMETER_STRING_VALUE_OR_USE_DEFAULT(fiber_file, config, "fibers_file");

    bool fibers_in_mesh = false;
    GET_PARAMETER_BOOLEAN_VALUE_OR_USE_DEFAULT(fibers_in_mesh, config, "fibers_in_mesh");

    struct fiber_coords *fibers = NULL;

    GET_PARAMETER_NUMERIC_VALUE_OR_REPORT_ERROR(real_cpu, sigma_l, config, "sigma_l");
    GET_PARAMETER_NUMERIC_VALUE_OR_REPORT_ERROR(real_cpu, sigma_t, config, "sigma_t");
    GET_PARAMETER_NUMERIC_VALUE_OR_REPORT_ERROR(real_cpu, sigma_n, config, "sigma_n");
    GET_PARAMETER_NUMERIC_VALUE_OR_USE_DEFAULT(real_cpu, sigma_C, config, "sigma_C");

    int number_layers_border_zone = 1;
    GET_PARAMETER_NUMERIC_VALUE_OR_USE_DEFAULT(int, number_layers_border_zone, config, "number_layers_bz");

    real sigma_factor = 1.0;
    GET_PARAMETER_NUMERIC_VALUE_OR_REPORT_ERROR(real, sigma_factor, config, "sigma_factor");

    real_cpu *f = NULL;
    real_cpu *s = NULL;
    real_cpu *n = NULL;

    if(fiber_file) {
        log_info("Loading mesh fibers\n");
        fibers = read_fibers(fiber_file, true);
    }
    else if(!fibers_in_mesh) {
        GET_PARAMETER_VECTOR_VALUE_OR_USE_DEFAULT(f, config, "f", 3);
        GET_PARAMETER_VECTOR_VALUE_OR_USE_DEFAULT(s, config, "s", 3);
        GET_PARAMETER_VECTOR_VALUE_OR_USE_DEFAULT(n, config, "n", 3);

        if(!f) {
            f = malloc(sizeof(real_cpu)*3);
            f[0] = 1.0;
            f[1] = 0.0;
            f[2] = 0.0;
        }

        if(!s) {
            s = malloc(sizeof(real_cpu)*3);
            s[0] = 0.0;
            s[1] = 1.0;
            s[2] = 0.0;
        }

        if(!n) {
            n = malloc(sizeof(real_cpu)*3);
            n[0] = 0.0;
            n[1] = 0.0;
            n[2] = 1.0;
        }

    }

    OMP(parallel for private(D))
    for(i = 0; i < num_active_cells; i++) {

        if(fibers) {
            int fiber_index = ac[i]->original_position_in_file;

            if(fiber_index == -1) {
                log_error_and_exit("fiber_index should not be -1, but it is for cell in index %d - %lf, %lf, %lf\n", i, ac[i]->center.x, ac[i]->center.y, ac[i]->center.z);
            }

            if(sigma_t == sigma_n) {
                calc_tensor2(D, fibers[fiber_index].f, sigma_l, sigma_t);
            }
            else {
                calc_tensor(D, fibers[fiber_index].f, fibers[fiber_index].s, fibers[fiber_index].n, sigma_l, sigma_t, sigma_n);
            }
            ac[i]->sigma.fibers = fibers[fiber_index];
        }
        else if(fibers_in_mesh) {
            if(sigma_t == sigma_n) {
                // log_info("%f, %f, %f, %f, %f, %f\n", ac[i]->center.x, ac[i]->center.y, ac[i]->center.z, ac[i]->sigma.fibers.f[0], ac[i]->sigma.fibers.f[1], ac[i]->sigma.fibers.f[2]);
                calc_tensor2(D, ac[i]->sigma.fibers.f, sigma_l, sigma_t);
            }
            else {
                calc_tensor(D, ac[i]->sigma.fibers.f, ac[i]->sigma.fibers.s, ac[i]->sigma.fibers.n, sigma_l, sigma_t, sigma_n);
            }

        }
        else {
            if(sigma_t == sigma_n) {
                calc_tensor2(D, f, sigma_l, sigma_t);
                ac[i]->sigma.fibers.f[0] = f[0];
                ac[i]->sigma.fibers.f[1] = f[1];
                ac[i]->sigma.fibers.f[2] = f[2];
            }
            else {
                calc_tensor(D, f, s, n, sigma_l, sigma_t, sigma_n);
                ac[i]->sigma.fibers.f[0] = f[0];
                ac[i]->sigma.fibers.f[1] = f[1];
                ac[i]->sigma.fibers.f[2] = f[2];

                ac[i]->sigma.fibers.s[0] = s[0];
                ac[i]->sigma.fibers.s[1] = s[1];
                ac[i]->sigma.fibers.s[2] = s[2];

                ac[i]->sigma.fibers.n[0] = n[0];
                ac[i]->sigma.fibers.n[1] = n[1];
                ac[i]->sigma.fibers.n[2] = n[2];
            }
        }

        ac[i]->sigma.x = D[0][0];
        ac[i]->sigma.y = D[1][1];
        ac[i]->sigma.z = D[2][2];

        ac[i]->sigma.xy = D[0][1];
        ac[i]->sigma.xz = D[0][2];
        ac[i]->sigma.yz = D[1][2];

        sigma_factor *= sigma_C;

        if(HU_FIBROTIC(ac[i])) {
            // log_info("%f, %f\n", sigma_factor, sigma_C);
            ac[i]->sigma.x *= sigma_factor;
            ac[i]->sigma.y *= sigma_factor;
            ac[i]->sigma.z *= sigma_factor;

            ac[i]->sigma.xy *= sigma_factor;
            ac[i]->sigma.xz *= sigma_factor;
            ac[i]->sigma.yz *= sigma_factor;
        }
        else
        {
            ac[i]->sigma.x *= sigma_C;
            ac[i]->sigma.y *= sigma_C;
            ac[i]->sigma.z *= sigma_C;

            ac[i]->sigma.xy *= sigma_C;
            ac[i]->sigma.xz *= sigma_C;
            ac[i]->sigma.yz *= sigma_C;
        }
        
        if (HU_BORDER_ZONE(ac[i])) {
            // log_info("eita, bz\n");
            ac[i]->sigma.x = ((ac[i]->sigma.x - sigma_factor * ac[i]->sigma.x) / number_layers_border_zone) * (HU_LAYER_BORDER_ZONE(ac[i])) + ac[i]->sigma.x * sigma_factor - ((ac[i]->sigma.x - ac[i]->sigma.x * sigma_factor) / (number_layers_border_zone));
            ac[i]->sigma.y = ((ac[i]->sigma.y - sigma_factor * ac[i]->sigma.y) / number_layers_border_zone) * (HU_LAYER_BORDER_ZONE(ac[i])) + ac[i]->sigma.y * sigma_factor - ((ac[i]->sigma.y - ac[i]->sigma.y * sigma_factor) / (number_layers_border_zone));
            ac[i]->sigma.z = ((ac[i]->sigma.z - sigma_factor * ac[i]->sigma.z) / number_layers_border_zone) * (HU_LAYER_BORDER_ZONE(ac[i])) + ac[i]->sigma.z * sigma_factor - ((ac[i]->sigma.z - ac[i]->sigma.z * sigma_factor) / (number_layers_border_zone));

            ac[i]->sigma.xy = ((ac[i]->sigma.xy - sigma_factor * ac[i]->sigma.xy) / number_layers_border_zone) * (HU_LAYER_BORDER_ZONE(ac[i])) + ac[i]->sigma.xy * sigma_factor - ((ac[i]->sigma.xy - ac[i]->sigma.xy * sigma_factor) / (number_layers_border_zone));
            ac[i]->sigma.xz = ((ac[i]->sigma.xz - sigma_factor * ac[i]->sigma.xz) / number_layers_border_zone) * (HU_LAYER_BORDER_ZONE(ac[i])) + ac[i]->sigma.xz * sigma_factor - ((ac[i]->sigma.xz - ac[i]->sigma.xz * sigma_factor) / (number_layers_border_zone));
            ac[i]->sigma.yz = ((ac[i]->sigma.yz - sigma_factor * ac[i]->sigma.yz) / number_layers_border_zone) * (HU_LAYER_BORDER_ZONE(ac[i])) + ac[i]->sigma.yz * sigma_factor - ((ac[i]->sigma.yz - ac[i]->sigma.yz * sigma_factor) / (number_layers_border_zone));
        } 
    }

    // OMP(parallel for)
    for(i = 0; i < num_active_cells; i++) {
        fill_discretization_matrix_elements_aniso(ac[i]);
    }

    free(f);
    free(s);
    free(n);
}
