#include <stdbool.h>
#include <stdlib.h>

#include <math.h>

#include "../3dparty/sds/sds.h"
#include "../alg/grid/grid.h"
#include "../config/save_mesh_config.h"
#include "../utils/utils.h"
#include "../extra_data_library/helper_functions.h"

#include "../domains_library/custom_mesh_info_data.h"
#include "../domains_library/mesh_info_data.h"
#include "../ensight_utils/ensight_grid.h"
#include "../vtk_utils/vtk_unstructured_grid.h"
#include "../monodomain/monodomain_solver.h"

#include "save_mesh_helper.h"

#ifdef COMPILE_CUDA
#include "../gpu_utils/gpu_utils.h"
#endif

static char *file_prefix;
static bool binary = false;
static bool clip_with_plain = false;
static bool clip_with_bounds = false;
static bool save_pvd = true;
static bool save_inactive = false;
static bool compress = false;
static bool save_f = false;
static int compression_level = 3;
char *output_dir_;
bool save_visible_mask_ = true;
bool save_scar_cells_ = false;
static bool initialized = false;
static bool save_ode_state_variables = false;

static void save_visibility_mask_(sds output_dir_with_file, ui8_array visible_cells) {
    sds output_dir_with_new_file = sdsnew(output_dir_with_file);
    output_dir_with_new_file = sdscat(output_dir_with_new_file, ".vis");
    FILE *vis = fopen(output_dir_with_new_file, "wb");
    fwrite(visible_cells, sizeof(uint8_t), arrlen(visible_cells), vis);
    sdsfree(output_dir_with_new_file);
    fclose(vis);
}

SAVE_MESH(save_as_text_or_binary_) {

    int iteration_count = time_info->iteration;

    real_cpu min_x = 0.0;
    real_cpu min_y = 0.0;
    real_cpu min_z = 0.0;
    real_cpu max_x = 0.0;
    real_cpu max_y = 0.0;
    real_cpu max_z = 0.0;

    float p0[3] = {1, 1, 1};
    float n[3] = {1, 1, 1};

//    if(!initialized) {
        GET_PARAMETER_STRING_VALUE_OR_REPORT_ERROR(output_dir_, config, "output_dir");
        GET_PARAMETER_STRING_VALUE_OR_REPORT_ERROR(file_prefix, config, "file_prefix");
        GET_PARAMETER_BOOLEAN_VALUE_OR_USE_DEFAULT(binary, config, "binary");
        GET_PARAMETER_BOOLEAN_VALUE_OR_USE_DEFAULT(clip_with_plain, config, "clip_with_plain");
        GET_PARAMETER_BOOLEAN_VALUE_OR_USE_DEFAULT(clip_with_bounds, config, "clip_with_bounds");
        GET_PARAMETER_BOOLEAN_VALUE_OR_USE_DEFAULT(save_inactive, config, "save_inactive_cells");
        GET_PARAMETER_BOOLEAN_VALUE_OR_USE_DEFAULT(save_visible_mask_, config, "save_visible_mask");
        GET_PARAMETER_BOOLEAN_VALUE_OR_USE_DEFAULT(save_ode_state_variables, config, "save_ode_state_variables");

        if(clip_with_plain) {
            GET_PARAMETER_NUMERIC_VALUE_OR_REPORT_ERROR(float, n[0], config, "normal_x");
            GET_PARAMETER_NUMERIC_VALUE_OR_REPORT_ERROR(float, n[1], config, "normal_y");
            GET_PARAMETER_NUMERIC_VALUE_OR_REPORT_ERROR(float, n[2], config, "normal_z");
            GET_PARAMETER_NUMERIC_VALUE_OR_REPORT_ERROR(float, p0[0], config, "origin_x");
            GET_PARAMETER_NUMERIC_VALUE_OR_REPORT_ERROR(float, p0[1], config, "origin_y");
            GET_PARAMETER_NUMERIC_VALUE_OR_REPORT_ERROR(float, p0[2], config, "origin_z");
        }

        if(clip_with_bounds) {
            GET_PARAMETER_NUMERIC_VALUE_OR_REPORT_ERROR(float, min_x, config, "min_x");
            GET_PARAMETER_NUMERIC_VALUE_OR_REPORT_ERROR(float, min_y, config, "min_y");
            GET_PARAMETER_NUMERIC_VALUE_OR_REPORT_ERROR(float, min_z, config, "min_z");
            GET_PARAMETER_NUMERIC_VALUE_OR_REPORT_ERROR(float, max_x, config, "max_x");
            GET_PARAMETER_NUMERIC_VALUE_OR_REPORT_ERROR(float, max_y, config, "max_y");
            GET_PARAMETER_NUMERIC_VALUE_OR_REPORT_ERROR(float, max_z, config, "max_z");
        }

        initialized = true;
 //   }

    real_cpu l = sqrtf(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    real_cpu A = n[0] / l;
    real_cpu B = n[1] / l;
    real_cpu C = n[2] / l;
    real_cpu D = -(n[0] * p0[0] + n[1] * p0[1] + n[2] * p0[2]);

    real_cpu side;

    sds tmp = sdsnew(output_dir_);
    tmp = sdscat(tmp, "/");

    sds base_name = NULL;
    if(binary) {
        base_name = create_base_name(file_prefix, iteration_count, "bin");
    } else {
        base_name = create_base_name(file_prefix, iteration_count, "txt");
    }

    tmp = sdscat(tmp, base_name);

    FILE *output_file = fopen(tmp, "w");

    struct cell_node *grid_cell = the_grid->first_cell;

    real_cpu center_x, center_y, center_z, dx, dy, dz;
    real_cpu v;

    ui8_array cell_visibility = NULL;
    arrsetcap(cell_visibility, the_grid->num_active_cells);

    while(grid_cell != 0) {

        if(grid_cell->active || save_inactive) {

            center_x = grid_cell->center.x;
            center_y = grid_cell->center.y;
            center_z = grid_cell->center.z;

            if(clip_with_plain) {
                side = A * center_x + B * center_y + C * center_z + D;
                if(side < 0) {
                    grid_cell = grid_cell->next;
                    continue;
                }
            }

            if(clip_with_bounds) {
                bool ignore_cell = center_x < min_x || center_x > max_x || center_y < min_y || center_y > max_y || center_z < min_z || center_z > max_z;

                if(ignore_cell) {
                    grid_cell = grid_cell->next;
                    continue;
                }
            }

            v = grid_cell->v;
            dx = grid_cell->discretization.x / 2.0;
            dy = grid_cell->discretization.y / 2.0;
            dz = grid_cell->discretization.z / 2.0;

            if(binary) {
                // TODO: maybe the size of the data should be always fixed (double as instance)
                fwrite(&center_x, sizeof(center_x), 1, output_file);
                fwrite(&center_y, sizeof(center_y), 1, output_file);
                fwrite(&center_z, sizeof(center_z), 1, output_file);
                fwrite(&dx, sizeof(dx), 1, output_file);
                fwrite(&dy, sizeof(dy), 1, output_file);
                fwrite(&dz, sizeof(dz), 1, output_file);
                fwrite(&v, sizeof(v), 1, output_file);
            } else {
                if(save_ode_state_variables) {

                    int n_state_vars = ode_solver->model_data.number_of_ode_equations - 1; // Vm is always saved
                    size_t num_sv_entries = n_state_vars + 1;
                    real *sv_cpu;

                    if(ode_solver->gpu) {

#ifdef COMPILE_CUDA
                        sv_cpu = MALLOC_ARRAY_OF_TYPE(real, ode_solver->original_num_cells * num_sv_entries);
                        check_cuda_error(cudaMemcpy2D(sv_cpu, ode_solver->original_num_cells * sizeof(real), ode_solver->sv, ode_solver->pitch,
                                    ode_solver->original_num_cells * sizeof(real), num_sv_entries, cudaMemcpyDeviceToHost));
#endif
                    } else {
                        sv_cpu = ode_solver->sv;
                    }

                    fprintf(output_file, "%g,%g,%g,%g,%g,%g,%g", center_x, center_y, center_z, dx, dy, dz, v);

                    for(int i = 1; i <= n_state_vars; i++) {
                        float value;
                        if(ode_solver->gpu) {
                            value = (float) sv_cpu[i*ode_solver->original_num_cells];
                        }
                        else {
                            value = sv_cpu[i];
                        }

                        fprintf(output_file, ",%g", value);
                    }

                    if(ode_solver->gpu) {
                        free(sv_cpu);
                    }

                    fprintf(output_file, "\n");
                }
                else {
                    fprintf(output_file, "%g,%g,%g,%g,%g,%g,%g\n", center_x, center_y, center_z, dx, dy, dz, v);
                }
            }
            arrput(cell_visibility, grid_cell->visible);
        }
        grid_cell = grid_cell->next;
    }

    if(save_visible_mask_) {
        save_visibility_mask_(tmp, cell_visibility);
    }

    sdsfree(base_name);
    sdsfree(tmp);

    fclose(output_file);

    CALL_EXTRA_FUNCTIONS(save_mesh_fn, time_info, config, the_grid, ode_solver, purkinje_ode_solver);

}

INIT_SAVE_MESH(init_save_as_vtk_or_vtu_) {
    if(config->persistent_data == NULL) {
        config->persistent_data = malloc(sizeof(struct common_persistent_data));
        ((struct common_persistent_data *)config->persistent_data)->grid = NULL;
        ((struct common_persistent_data *)config->persistent_data)->first_save_call = true;
    }
}

END_SAVE_MESH(end_save_as_vtk_or_vtu_) {
    free_vtk_unstructured_grid(((struct common_persistent_data *)config->persistent_data)->grid);
    free(config->persistent_data);
    config->persistent_data = NULL;
}

SAVE_MESH(save_as_vtk_) {

    int iteration_count = time_info->iteration;

    if(((struct common_persistent_data *)config->persistent_data)->first_save_call) {
        GET_PARAMETER_STRING_VALUE_OR_REPORT_ERROR(output_dir_, config, "output_dir");
        GET_PARAMETER_STRING_VALUE_OR_REPORT_ERROR(file_prefix, config, "file_prefix");
        GET_PARAMETER_BOOLEAN_VALUE_OR_USE_DEFAULT(clip_with_plain, config, "clip_with_plain");
        GET_PARAMETER_BOOLEAN_VALUE_OR_USE_DEFAULT(clip_with_bounds, config, "clip_with_bounds");
        GET_PARAMETER_BOOLEAN_VALUE_OR_USE_DEFAULT(binary, config, "binary");
        GET_PARAMETER_BOOLEAN_VALUE_OR_USE_DEFAULT(save_f, config, "save_f");
        GET_PARAMETER_BOOLEAN_VALUE_OR_USE_DEFAULT(save_visible_mask_, config, "save_visible_mask");
        GET_PARAMETER_BOOLEAN_VALUE_OR_USE_DEFAULT(save_scar_cells_, config, "save_scar_cells");

        ((struct common_persistent_data *)config->persistent_data)->first_save_call = false;
    }
    float plain_coords[6] = {0, 0, 0, 0, 0, 0};
    float bounds[6] = {0, 0, 0, 0, 0, 0};

    if(clip_with_plain) {
        GET_PARAMETER_NUMERIC_VALUE_OR_REPORT_ERROR(float, plain_coords[0], config, "origin_x");
        GET_PARAMETER_NUMERIC_VALUE_OR_REPORT_ERROR(float, plain_coords[1], config, "origin_y");
        GET_PARAMETER_NUMERIC_VALUE_OR_REPORT_ERROR(float, plain_coords[2], config, "origin_z");
        GET_PARAMETER_NUMERIC_VALUE_OR_REPORT_ERROR(float, plain_coords[3], config, "normal_x");
        GET_PARAMETER_NUMERIC_VALUE_OR_REPORT_ERROR(float, plain_coords[4], config, "normal_y");
        GET_PARAMETER_NUMERIC_VALUE_OR_REPORT_ERROR(float, plain_coords[5], config, "normal_z");
    }

    if(clip_with_bounds) {
        GET_PARAMETER_NUMERIC_VALUE_OR_REPORT_ERROR(float, bounds[0], config, "min_x");
        GET_PARAMETER_NUMERIC_VALUE_OR_REPORT_ERROR(float, bounds[1], config, "min_y");
        GET_PARAMETER_NUMERIC_VALUE_OR_REPORT_ERROR(float, bounds[2], config, "min_z");
        GET_PARAMETER_NUMERIC_VALUE_OR_REPORT_ERROR(float, bounds[3], config, "max_x");
        GET_PARAMETER_NUMERIC_VALUE_OR_REPORT_ERROR(float, bounds[4], config, "max_y");
        GET_PARAMETER_NUMERIC_VALUE_OR_REPORT_ERROR(float, bounds[5], config, "max_z");
    }

    sds output_dir_with_file = sdsnew(output_dir_);
    output_dir_with_file = sdscat(output_dir_with_file, "/");
    sds base_name = create_base_name(file_prefix, iteration_count, "vtk");

    real_cpu current_t = time_info->current_t;

    // TODO: change this. We dont need the current_t here
    output_dir_with_file = sdscatprintf(output_dir_with_file, base_name, current_t);

    bool read_only_data = ((struct common_persistent_data *)config->persistent_data)->grid != NULL;

    new_vtk_unstructured_grid_from_alg_grid(&(((struct common_persistent_data *)config->persistent_data)->grid), the_grid, clip_with_plain,
                                            plain_coords, clip_with_bounds, bounds, read_only_data, save_f, save_scar_cells_, NULL);

    save_vtk_unstructured_grid_as_legacy_vtk(((struct common_persistent_data *)config->persistent_data)->grid, output_dir_with_file, binary,
                                             save_f, NULL);

    if(save_visible_mask_) {
        save_visibility_mask_(output_dir_with_file, (((struct common_persistent_data *)config->persistent_data)->grid)->cell_visibility);
    }

    if(the_grid->adaptive) {
        free_vtk_unstructured_grid(((struct common_persistent_data *)config->persistent_data)->grid);
        ((struct common_persistent_data *)config->persistent_data)->grid = NULL;
    }

    sdsfree(output_dir_with_file);
    sdsfree(base_name);

    CALL_EXTRA_FUNCTIONS(save_mesh_fn, time_info, config, the_grid, ode_solver, purkinje_ode_solver);
}

SAVE_MESH(save_as_vtu_) {

    int iteration_count = time_info->iteration;

    if(((struct common_persistent_data *)config->persistent_data)->first_save_call) {
        GET_PARAMETER_STRING_VALUE_OR_REPORT_ERROR(output_dir_, config, "output_dir");
        GET_PARAMETER_STRING_VALUE_OR_REPORT_ERROR(file_prefix, config, "file_prefix");
        GET_PARAMETER_BOOLEAN_VALUE_OR_USE_DEFAULT(clip_with_plain, config, "clip_with_plain");
        GET_PARAMETER_BOOLEAN_VALUE_OR_USE_DEFAULT(clip_with_bounds, config, "clip_with_bounds");
        GET_PARAMETER_BOOLEAN_VALUE_OR_USE_DEFAULT(binary, config, "binary");
        GET_PARAMETER_BOOLEAN_VALUE_OR_USE_DEFAULT(save_pvd, config, "save_pvd");
        GET_PARAMETER_BOOLEAN_VALUE_OR_USE_DEFAULT(compress, config, "compress");
        GET_PARAMETER_NUMERIC_VALUE_OR_USE_DEFAULT(int, compression_level, config, "compression_level");
        GET_PARAMETER_BOOLEAN_VALUE_OR_USE_DEFAULT(save_visible_mask_, config, "save_visible_mask");
        GET_PARAMETER_BOOLEAN_VALUE_OR_USE_DEFAULT(save_scar_cells_, config, "save_scar_cells");

        if(compress)
            binary = true;

        if(!save_pvd) {
            ((struct common_persistent_data *)config->persistent_data)->first_save_call = false;
        }
    }

    float plain_coords[6] = {0, 0, 0, 0, 0, 0};
    float bounds[6] = {0, 0, 0, 0, 0, 0};

    if(clip_with_plain) {
        GET_PARAMETER_NUMERIC_VALUE_OR_REPORT_ERROR(float, plain_coords[0], config, "origin_x");
        GET_PARAMETER_NUMERIC_VALUE_OR_REPORT_ERROR(float, plain_coords[1], config, "origin_y");
        GET_PARAMETER_NUMERIC_VALUE_OR_REPORT_ERROR(float, plain_coords[2], config, "origin_z");
        GET_PARAMETER_NUMERIC_VALUE_OR_REPORT_ERROR(float, plain_coords[3], config, "normal_x");
        GET_PARAMETER_NUMERIC_VALUE_OR_REPORT_ERROR(float, plain_coords[4], config, "normal_y");
        GET_PARAMETER_NUMERIC_VALUE_OR_REPORT_ERROR(float, plain_coords[5], config, "normal_z");
    }

    if(clip_with_bounds) {
        GET_PARAMETER_NUMERIC_VALUE_OR_REPORT_ERROR(float, bounds[0], config, "min_x");
        GET_PARAMETER_NUMERIC_VALUE_OR_REPORT_ERROR(float, bounds[1], config, "min_y");
        GET_PARAMETER_NUMERIC_VALUE_OR_REPORT_ERROR(float, bounds[2], config, "min_z");
        GET_PARAMETER_NUMERIC_VALUE_OR_REPORT_ERROR(float, bounds[3], config, "max_x");
        GET_PARAMETER_NUMERIC_VALUE_OR_REPORT_ERROR(float, bounds[4], config, "max_y");
        GET_PARAMETER_NUMERIC_VALUE_OR_REPORT_ERROR(float, bounds[5], config, "max_z");
    }

    sds output_dir_with_file = sdsnew(output_dir_);
    output_dir_with_file = sdscat(output_dir_with_file, "/");
    sds base_name = create_base_name(file_prefix, iteration_count, "vtu");

    real_cpu current_t = time_info->current_t;

    output_dir_with_file = sdscatprintf(output_dir_with_file, base_name, current_t);

    if(save_pvd) {
        add_file_to_pvd(current_t, output_dir_, base_name, ((struct common_persistent_data *)config->persistent_data)->first_save_call);
        ((struct common_persistent_data *)config->persistent_data)->first_save_call = false;
    }

    bool read_only_data = ((struct common_persistent_data *)config->persistent_data)->grid != NULL;
    new_vtk_unstructured_grid_from_alg_grid(&((struct common_persistent_data *)config->persistent_data)->grid, the_grid, clip_with_plain,
                                            plain_coords, clip_with_bounds, bounds, read_only_data, save_f, save_scar_cells_, NULL);

    if(compress) {
        save_vtk_unstructured_grid_as_vtu_compressed(((struct common_persistent_data *)config->persistent_data)->grid, output_dir_with_file,
                                                     compression_level);
    } else {
        save_vtk_unstructured_grid_as_vtu(((struct common_persistent_data *)config->persistent_data)->grid, output_dir_with_file, binary);
    }

    if(save_visible_mask_) {
        save_visibility_mask_(output_dir_with_file, (((struct common_persistent_data *)config->persistent_data)->grid)->cell_visibility);
    }

    sdsfree(output_dir_with_file);
    sdsfree(base_name);

    // TODO: I do not know if we should to this here or call the end and init save functions on the adaptivity step.....
    if(the_grid->adaptive) {
        free_vtk_unstructured_grid(((struct common_persistent_data *)config->persistent_data)->grid);
        ((struct common_persistent_data *)config->persistent_data)->grid = NULL;
    }

    CALL_EXTRA_FUNCTIONS(save_mesh_fn, time_info, config, the_grid, ode_solver, purkinje_ode_solver);

}

INIT_SAVE_MESH(init_save_as_ensight_) {
    if(config->persistent_data == NULL) {
        config->persistent_data = calloc(1, sizeof(struct common_persistent_data));
    }
}

SAVE_MESH(save_as_ensight_) {

    struct common_persistent_data *persistent_data = (struct common_persistent_data*) config->persistent_data;

    if(the_grid == NULL) {
        log_error_and_exit("Error in save_as_ensight. No grid defined\n");
    }

    if(the_grid->num_active_cells == 0 && the_grid->purkinje == NULL) {
        log_error_and_exit("Error in save_as_ensight. No grid and/or no purkinje grid defined\n");
    }

    if(the_grid->adaptive) {
        log_error_and_exit("save_as_ensight function does not support adaptive meshes yet! Aborting\n");
    }

    static int n_state_vars = 0;
    static bool geometry_saved = false;
    static uint32_t num_files = 0;

    if(!geometry_saved) {

        int print_rate = 1;

        char *mesh_format = NULL;
        GET_PARAMETER_STRING_VALUE_OR_USE_DEFAULT(mesh_format, config, "mesh_format");

        //We are getting called from save_with_activation_times
        if(mesh_format != NULL) {
            GET_PARAMETER_NUMERIC_VALUE_OR_USE_DEFAULT(int, print_rate, config, "mesh_print_rate");
        }
        else { //We are being directly called
            GET_PARAMETER_NUMERIC_VALUE_OR_USE_DEFAULT(int, print_rate, config, "print_rate");
        }

        GET_PARAMETER_STRING_VALUE_OR_REPORT_ERROR(output_dir_, config, "output_dir");
        GET_PARAMETER_BOOLEAN_VALUE_OR_USE_DEFAULT(binary, config, "binary");
        GET_PARAMETER_BOOLEAN_VALUE_OR_USE_DEFAULT(save_visible_mask_, config, "save_visible_mask");
        GET_PARAMETER_BOOLEAN_VALUE_OR_USE_DEFAULT(save_ode_state_variables, config, "save_ode_state_variables");

        num_files = ((time_info->final_t / time_info->dt) / print_rate) + 1;

        sds output_dir_with_file = sdsnew(output_dir_);
        output_dir_with_file = sdscat(output_dir_with_file, "/geometry.geo");

        struct ensight_grid *ensight_grid = new_ensight_grid_from_alg_grid(the_grid, false, NULL, false, NULL, false, false);
        save_ensight_grid_as_ensight6_geometry(ensight_grid, output_dir_with_file, binary);

        if(save_visible_mask_) {
            save_visibility_mask_(output_dir_with_file, ensight_grid->parts[0].cell_visibility);
        }

        free_ensight_grid(ensight_grid);

        sdsfree(output_dir_with_file);

        output_dir_with_file = sdsnew(output_dir_);
        output_dir_with_file = sdscat(output_dir_with_file, "/simulation_result.case");

        if(save_ode_state_variables) {
            n_state_vars = ode_solver->model_data.number_of_ode_equations - 1; // Vm is always saved
        }

        save_case_file(output_dir_with_file, num_files, time_info->dt, print_rate, n_state_vars);

        sdsfree(output_dir_with_file);
        geometry_saved = true;
    }

    sds output_dir_with_file = sdsnew(output_dir_);
    output_dir_with_file = sdscat(output_dir_with_file, "/");

    if(persistent_data->n_digits == 0) {
        persistent_data->n_digits = log10(num_files*500) + 1;
    }

    sds base_name = sdscatprintf(sdsempty(), "Vm.Esca%%0%dd", persistent_data->n_digits);

    char tmp[8192];
    sprintf(tmp, base_name, persistent_data->file_count);

    output_dir_with_file = sdscatprintf(output_dir_with_file, "/%s", tmp);

    save_en6_result_file(output_dir_with_file, the_grid, binary);

    sdsfree(base_name);
    sdsfree(output_dir_with_file);

    if(n_state_vars) {
        size_t num_sv_entries = ode_solver->model_data.number_of_ode_equations;
        base_name = sdscatprintf(sdsempty(), "Sv%%d.Esca%%0%dd", persistent_data->n_digits);
        real *sv_cpu;

        if(ode_solver->gpu) {

#ifdef COMPILE_CUDA
            sv_cpu = MALLOC_ARRAY_OF_TYPE(real, ode_solver->original_num_cells * num_sv_entries);
            check_cuda_error(cudaMemcpy2D(sv_cpu, ode_solver->original_num_cells * sizeof(real), ode_solver->sv, ode_solver->pitch,
                                          ode_solver->original_num_cells * sizeof(real), num_sv_entries, cudaMemcpyDeviceToHost));
#endif
        } else {
            sv_cpu = ode_solver->sv;
        }

        for(int i = 1; i <= n_state_vars; i++) {

            char tmp[8192];
            sprintf(tmp, base_name, i, persistent_data->file_count);

            sds output_dir_with_file = sdsnew(output_dir_);
            output_dir_with_file = sdscat(output_dir_with_file, "/");

            output_dir_with_file = sdscatprintf(output_dir_with_file, "/%s", tmp);

            save_en6_result_file_state_vars(output_dir_with_file, sv_cpu, ode_solver->original_num_cells, num_sv_entries, i, binary, ode_solver->gpu);
            sdsfree(output_dir_with_file);
        }

        sdsfree(base_name);

        if(ode_solver->gpu) {
            free(sv_cpu);
        }
    }

    persistent_data->file_count++;

    CALL_EXTRA_FUNCTIONS(save_mesh_fn, time_info, config, the_grid, ode_solver, purkinje_ode_solver);
}

END_SAVE_MESH(end_save_as_ensight_) {
    free(config->persistent_data);
    config->persistent_data = NULL;
}

static bool update_ode_state_vector_and_check_for_activity_(real_cpu vm_threshold, struct ode_solver *the_ode_solver, struct ode_solver *the_purkinje_ode_solver,
                                                    struct grid *the_grid) {
    bool act = false;

    // Tissue section
    uint32_t n_active = the_grid->num_active_cells;
    struct cell_node **ac = the_grid->active_cells;

    if(the_ode_solver) {
        int n_odes = the_ode_solver->model_data.number_of_ode_equations;

        real *sv = the_ode_solver->sv;

        if(the_ode_solver->gpu) {
#ifdef COMPILE_CUDA
            uint32_t max_number_of_cells = the_ode_solver->original_num_cells;
            real *vms;
            size_t mem_size = max_number_of_cells * sizeof(real);

            vms = (real *)malloc(mem_size);

            if(the_grid->adaptive)
                check_cuda_error(cudaMemcpy(vms, sv, mem_size, cudaMemcpyDeviceToHost));

            OMP(parallel for)
            for(uint32_t i = 0; i < n_active; i++) {
                vms[ac[i]->sv_position] = (real)ac[i]->v;

                if(ac[i]->v > vm_threshold) {
                    act = true;
                }
            }

            check_cuda_error(cudaMemcpy(sv, vms, mem_size, cudaMemcpyHostToDevice));
            free(vms);
#endif
        } else {
            OMP(parallel for)
            for(uint32_t i = 0; i < n_active; i++) {
                sv[ac[i]->sv_position * n_odes] = (real)ac[i]->v;

                if(ac[i]->v > vm_threshold) {
                    act = true;
                }
            }
        }
    }

    if(the_purkinje_ode_solver) {
        // Purkinje section
        uint32_t n_active_purkinje = the_grid->purkinje->number_of_purkinje_cells;
        struct cell_node **ac_purkinje = the_grid->purkinje->purkinje_cells;

        int n_odes_purkinje = the_purkinje_ode_solver->model_data.number_of_ode_equations;

        real *sv_purkinje = the_purkinje_ode_solver->sv;

        if(the_purkinje_ode_solver->gpu) {
#ifdef COMPILE_CUDA
            uint32_t max_number_of_purkinje_cells = the_purkinje_ode_solver->original_num_cells;
            real *vms_purkinje;
            size_t mem_size_purkinje = max_number_of_purkinje_cells * sizeof(real);

            vms_purkinje = (real *)malloc(mem_size_purkinje);

            if(the_grid->adaptive)
                check_cuda_error(cudaMemcpy(vms_purkinje, sv_purkinje, mem_size_purkinje, cudaMemcpyDeviceToHost));

            OMP(parallel for)
            for(uint32_t i = 0; i < n_active_purkinje; i++) {
                vms_purkinje[ac_purkinje[i]->sv_position] = (real)ac_purkinje[i]->v;

                if(ac_purkinje[i]->v > vm_threshold) {
                    act = true;
                }
            }

            check_cuda_error(cudaMemcpy(sv_purkinje, vms_purkinje, mem_size_purkinje, cudaMemcpyHostToDevice));
            free(vms_purkinje);
#endif
        } else {
            OMP(parallel for)
            for(uint32_t i = 0; i < n_active_purkinje; i++) {
                sv_purkinje[ac_purkinje[i]->sv_position * n_odes_purkinje] = (real)ac_purkinje[i]->v;

                if(ac_purkinje[i]->v > vm_threshold) {
                    act = true;
                }
            }
        }
    }

    return act;
}

INIT_SAVE_MESH(init_save_ecg_data) {
    if (config->persistent_data == NULL) {
        config->persistent_data = calloc(1, sizeof(struct common_persistent_data));
        struct common_persistent_data *cpd = (struct common_persistent_data *) config->persistent_data;
        cpd->first_save_call = true;
    }
}

END_SAVE_MESH(end_save_ecg_data) {
    free(config->persistent_data);
    config->persistent_data = NULL;
}

SAVE_MESH(save_ecg_data) {
    struct common_persistent_data *cpd = (struct common_persistent_data *) config->persistent_data;

    // Diretório de saída
    GET_PARAMETER_STRING_VALUE_OR_REPORT_ERROR(output_dir_, config, "output_dir");

    // Obtenção do tempo atual
    real_cpu current_t = time_info->current_t;

    // Nome do arquivo estático (salvo apenas uma vez)
    static bool static_data_saved = false;
    if (!static_data_saved) {
        sds static_file_path = sdsnew(output_dir_);
        static_file_path = sdscat(static_file_path, "/ecg_static_data.txt");

        FILE *static_file = fopen(static_file_path, "w");
        if (!static_file) {
            log_error_and_exit("save_ecg_data - Unable to open file %s!\n", static_file_path);
        }

        // Obter parâmetros globais
        real_cpu sigma_b = 20.0; // Valor padrão caso não encontrado
        char ini_path[512];
        snprintf(ini_path, sizeof(ini_path), "%s/original_configuration.ini", output_dir_);

        FILE *file = fopen(ini_path, "r");
        if (file == NULL) {
            fprintf(stderr, "Could not open original_configuration.ini\n");
            sigma_b = 1.0; // Valor padrão
        } else {
            char line[256];
            int in_calc_ecg_section = 0;

            while (fgets(line, sizeof(line), file)) {
                line[strcspn(line, "\n")] = 0;

                if (strcmp(line, "[calc_ecg]") == 0) {
                    in_calc_ecg_section = 1;
                    continue;
                }

                if (in_calc_ecg_section && line[0] == '[') {
                    break;
                }

                if (in_calc_ecg_section && strstr(line, "sigma_b") == line) {
                    sscanf(line, "sigma_b = %lf", &sigma_b);
                    break;
                }
            }
            fclose(file);
        }

        // Salvar informações globais
        fprintf(static_file, "# Scale Factor: %lf\n", 1.0 / (4.0 * M_PI * sigma_b));

        // Obter e salvar os leads
        double *lead_x = NULL, *lead_y = NULL, *lead_z = NULL;
        int n_leads = 0;

        file = fopen(ini_path, "r");
        if (file) {
            char line[256];
            int in_calc_ecg_section = 0;

            while (fgets(line, sizeof(line), file)) {
                line[strcspn(line, "\n")] = 0;

                if (strcmp(line, "[calc_ecg]") == 0) {
                    in_calc_ecg_section = 1;
                    continue;
                }

                if (in_calc_ecg_section && line[0] == '[') {
                    break;
                }

                if (in_calc_ecg_section && strstr(line, "lead") == line) {
                    lead_x = realloc(lead_x, (n_leads + 1) * sizeof(double));
                    lead_y = realloc(lead_y, (n_leads + 1) * sizeof(double));
                    lead_z = realloc(lead_z, (n_leads + 1) * sizeof(double));

                    sscanf(line, "lead%*d = %lf,%lf,%lf", &lead_x[n_leads], &lead_y[n_leads], &lead_z[n_leads]);
                    n_leads++;
                }
            }
            fclose(file);
        }

        fprintf(static_file, "# Leads (x, y, z):\n");
        for (int i = 0; i < n_leads; i++) {
            fprintf(static_file, "%lf %lf %lf\n", lead_x[i], lead_y[i], lead_z[i]);
        }

        free(lead_x);
        free(lead_y);
        free(lead_z);

        // Salvar informações das células
        fprintf(static_file, "# Cell Data (center_x, center_y, center_z, dx, dy, dz):\n");
        struct cell_node *grid_cell = the_grid->first_cell;

        while (grid_cell != 0) {
            if (grid_cell->active) {
                real_cpu center_x = grid_cell->center.x;
                real_cpu center_y = grid_cell->center.y;
                real_cpu center_z = grid_cell->center.z;

                real_cpu dx = grid_cell->discretization.x / 2.0;
                real_cpu dy = grid_cell->discretization.y / 2.0;
                real_cpu dz = grid_cell->discretization.z / 2.0;

                fprintf(static_file, "%lf %lf %lf %lf %lf %lf\n", center_x, center_y, center_z, dx, dy, dz);
            }
            grid_cell = grid_cell->next;
        }

        fclose(static_file);
        sdsfree(static_file_path);
        static_data_saved = true;
    }

    // Nome do arquivo dinâmico (por timestamp)
    sds dynamic_file_path = sdsnew(output_dir_);
    dynamic_file_path = sdscatprintf(dynamic_file_path, "/ecg_beta_im_t=%lf.bin", current_t);

    FILE *dynamic_file = fopen(dynamic_file_path, "w");
    if (!dynamic_file) {
        log_error_and_exit("save_ecg_data - Unable to open file %s!\n", dynamic_file_path);
    }

    real_cpu cM = 1.0; // Capacitância da membrana
    real_cpu delta_t = 0.02; // Intervalo de tempo (pode ser alterado posteriormente)
    real_cpu beta = 0.14; // Intervalo de tempo (pode ser alterado posteriormente)

    // Iterar pelas células e salvar os valores de \( \beta_{im} \)
    struct cell_node *grid_cell = the_grid->first_cell;
    while (grid_cell != 0) {
        if (grid_cell->active) {
            // Obter índice da célula ativa
            // Calcular o volume da célula
    real_cpu dx = grid_cell->discretization.x;
    real_cpu dy = grid_cell->discretization.y;
    real_cpu dz = grid_cell->discretization.z;
    real_cpu volume = dx * dy * dz;

    // Obter os elementos da célula (já montados considerando orientação/fibra)
    struct element *cell_elements = grid_cell->elements;
    size_t max_el = arrlen(cell_elements);

    // Calcular beta_im como no ecg.c (considerando orientação)
    real_cpu beta_im = 0.0;
    for(size_t el = 0; el < max_el; el++) {
        beta_im += cell_elements[el].value_ecg * cell_elements[el].cell->v;
    }
    beta_im = beta_im / volume;

    // Salvar o beta_im
    fwrite(&beta_im, sizeof(real_cpu), 1, dynamic_file);

        }

        // Ir para a próxima célula
        grid_cell = grid_cell->next;
    }


    fclose(dynamic_file);
    sdsfree(dynamic_file_path);
}