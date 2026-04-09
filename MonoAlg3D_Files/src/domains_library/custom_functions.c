//
// Created by sachetto on 16/03/2021.
//

#include "domain_helpers.h"
#include "custom_mesh_info_data.h"
#include "mesh_info_data.h"

#include "../3dparty/sds/sds.h"
#include "../3dparty/stb_ds.h"
#include "../config_helpers/config_helpers.h"
#include "../libraries_common/common_data_structures.h"
#include "../logger/logger.h"
#include "../utils/utils.h"
#include <time.h>
#include <unistd.h>
#include "../utils/stop_watch.h"
#include <math.h>
#include <float.h>

SET_CUSTOM_DATA_FOR_MESH(set_custom_data_for_2D_mesh) {

    INITIALIZE_FIBROTIC_INFO(cell);
    
    FIBROTIC(cell) = (custom_data[0] == 1);

    // informacao do u
    if (custom_data[1] < 0.3) {
       TISSUE_TYPE(cell) = 2; // EPI
    }
    else if (custom_data[1] < 0.55) {
       TISSUE_TYPE(cell) = 1; // MID
    }
    else {
      TISSUE_TYPE(cell) = 0; // ENDO
    }

    cell->sigma.fibers.f[0] = custom_data[2];
    cell->sigma.fibers.f[1] = custom_data[3];
    cell->sigma.fibers.f[2] = custom_data[4];
}

SET_CUSTOM_DATA_FOR_MESH(set_custom_data_for_hu_mesh_with_fibers_v3) {
	
	//Ordem .alg:
	//tecido
	//fenotipo
	//f
	//s
	//n

    HU_INITIALIZE_FIBROTIC_INFO(cell);
    
    HU_FIBROTIC(cell) = (custom_data[0] == 1);
    HU_BORDER_ZONE(cell) = (custom_data[0] != 1 && custom_data[0] != 0);

    if (HU_BORDER_ZONE(cell)){
        HU_LAYER_BORDER_ZONE(cell) = custom_data[0];
    }
    
    // informacao do u
    if (custom_data[1] < 0.3) {
       HU_TISSUE_TYPE(cell) = 2; // EPI
    }
    else if (custom_data[1] < 0.55) {
       HU_TISSUE_TYPE(cell) = 1; // MID
    }
    else {
       HU_TISSUE_TYPE(cell) = 0; // ENDO
    }
    
    cell->sigma.fibers.f[0] = custom_data[2];
    cell->sigma.fibers.f[1] = custom_data[3];
    cell->sigma.fibers.f[2] = custom_data[4];
    
    cell->sigma.fibers.s[0] = custom_data[5];
    cell->sigma.fibers.s[1] = custom_data[6];
    cell->sigma.fibers.s[2] = custom_data[7];
    
    cell->sigma.fibers.n[0] = custom_data[8];
    cell->sigma.fibers.n[1] = custom_data[9];
    cell->sigma.fibers.n[2] = custom_data[10];
}

SET_SPATIAL_DOMAIN(initialize_grid_with_hu_mesh_with_scar) {

    char *mesh_file = NULL;
    GET_PARAMETER_STRING_VALUE_OR_REPORT_ERROR(mesh_file, config, "mesh_file");

    real_cpu start_h = 100.0;
    GET_PARAMETER_NUMERIC_VALUE_OR_USE_DEFAULT(real_cpu, start_h, config, "original_discretization");

    real_cpu desired_h = 100.0;
    GET_PARAMETER_NUMERIC_VALUE_OR_USE_DEFAULT(real_cpu, desired_h, config, "desired_discretization");

    assert(the_grid);

    log_info("Loading HU Heart Mesh with discretization: %lf\n", desired_h);

    uint32_t num_volumes = 514389;
    GET_PARAMETER_NUMERIC_VALUE_OR_USE_DEFAULT(uint32_t, num_volumes, config, "num_volumes");

    uint32_t num_extra_fields = 1;
    GET_PARAMETER_NUMERIC_VALUE_OR_USE_DEFAULT(uint32_t, num_extra_fields, config, "num_extra_fields");

    int num_loaded = 0;

    if(num_extra_fields == 1)
        num_loaded = set_custom_mesh_from_file(the_grid, mesh_file, num_volumes, start_h, num_extra_fields, set_custom_data_for_hu_mesh);
    else if (num_extra_fields == 11)
		num_loaded = set_custom_mesh_from_file(the_grid, mesh_file, num_volumes, start_h, num_extra_fields, set_custom_data_for_hu_mesh_with_fibers_v3);
    else
        num_loaded = set_custom_mesh_from_file(the_grid, mesh_file, num_volumes, start_h, num_extra_fields, set_custom_data_for_2D_mesh);

    log_info("Read %d volumes from file (expected %d): %s\n", num_loaded, num_volumes, mesh_file);

    int num_refs_remaining = (int)(start_h / desired_h) - 1;

    if(num_refs_remaining > 0) {
        refine_grid(the_grid, num_refs_remaining);
    }

    the_grid->start_discretization = SAME_POINT3D(desired_h);

    free(mesh_file);

    return num_loaded;
}