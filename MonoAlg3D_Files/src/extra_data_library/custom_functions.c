#include <unistd.h>

#include "../config/extra_data_config.h"
#include "../config_helpers/config_helpers.h"
#include "../libraries_common/common_data_structures.h"
#include "../domains_library/mesh_info_data.h"
#include "../domains_library/custom_mesh_info_data.h"
#include "helper_functions.h"
#include "../logger/logger.h"
#include <time.h>
#include <unistd.h>

SET_EXTRA_DATA (set_extra_data_mixed_tt3_endo) {
    uint32_t num_active_cells = the_grid->num_active_cells;
    real side_length = the_grid->mesh_side_length.x;
    struct cell_node ** ac = the_grid->active_cells;

    struct extra_data_for_tt3 *extra_data = NULL;
    extra_data = set_common_tt3_data(config, num_active_cells);
	
	int i;

    // Transmurality and fibrosis tags
	OMP(parallel for)
    for (int i = 0; i < num_active_cells; i++) {

        real center_x = ac[i]->center.x;

        // Tag the model transmurality
        // ENDO=0, MID=1, EPI=2
        extra_data->transmurality[i] = 0.0;

        // Tag the fibrosis region
        extra_data->fibrosis[i] = 1.0;
    }

    SET_EXTRA_DATA_SIZE(sizeof(struct extra_data_for_tt3));

    return (void*)extra_data;
}