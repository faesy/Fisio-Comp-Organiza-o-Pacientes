#ifndef __CUSTOM_MESH_INFO_DATA_H
#define __CUSTOM_MESH_INFO_DATA_H

#include <stdbool.h>
#include "../common_types/common_types.h"
#include "mesh_info_data.h"

struct dhzb_mesh_info {

    enum tissue_type {
        SCAR,
        BZ,
        HEALTH
    } tissue_type;

    enum location {
        BIG_SCAR,
        SMALL_SCAR,
        HEALTH_AREA
    } location;
};

struct hcm_mesh_info {
    real_cpu tissue_type;
    bool septum;
    uint32_t closest_element_id;
    uint32_t closest_node_id;
};

#define DHZB_MESH_INFO(grid_cell) (struct dhzb_mesh_info *)grid_cell->mesh_extra_info
#define DHZB_MESH_TISSUE_TYPE(grid_cell) (DHZB_MESH_INFO(grid_cell))->tissue_type
#define DHZB_MESH_LOCATION(grid_cell) (DHZB_MESH_INFO(grid_cell))->location

#define INITIALIZE_DHZB_MESH_INFO(grid_cell)                                                                           \
    do {                                                                                                               \
        ALLOCATE_MESH_INFO(grid_cell, dhzb_mesh_info);                                                                 \
        DHZB_MESH_TISSUE_TYPE ((grid_cell)) = HEALTH;                                                                  \
        DHZB_MESH_LOCATION (grid_cell) = HEALTH_AREA;                                                                  \
} while (0)


#define INITIALIZE_SCV_INFO(grid_cell) INITIALIZE_FIBROTIC_INFO(grid_cell)

#define HCM_INFO(grid_cell) (struct hcm_mesh_info *)grid_cell->mesh_extra_info
#define HCM_SEPTUM(grid_cell) (HCM_INFO(grid_cell))->septum
#define HCM_TISSUE_TYPE(grid_cell) (HCM_INFO(grid_cell))->tissue_type
#define HCM_ELEMENT_ID(grid_cell) (HCM_INFO(grid_cell))->closest_element_id
#define HCM_NODE_ID(grid_cell) (HCM_INFO(grid_cell))->closest_node_id

#define INITIALIZE_HCM_INFO(grid_cell)                                                                                 \
    do {                                                                                                               \
        size_t __size__ = sizeof (struct hcm_mesh_info);                                                               \
        (grid_cell)->mesh_extra_info = malloc (__size__);                                                              \
        (grid_cell)->mesh_extra_info_size = __size__;                                                                  \
        HCM_SEPTUM ((grid_cell)) = false;                                                                              \
        HCM_TISSUE_TYPE ((grid_cell)) = 0;                                                                             \
} while (0)

struct fibrotic_and_layered_border_zone_mesh_info {
    bool fibrotic;
    bool border_zone;
    int tissue_type;
    int layer_border_zone;
};

#define HU_FIBROTIC_INFO(grid_cell) (struct fibrotic_and_layered_border_zone_mesh_info *)(grid_cell)->mesh_extra_info
#define HU_FIBROTIC(grid_cell) (HU_FIBROTIC_INFO(grid_cell))->fibrotic
#define HU_BORDER_ZONE(grid_cell) (HU_FIBROTIC_INFO(grid_cell))->border_zone
#define HU_TISSUE_TYPE(grid_cell) (HU_FIBROTIC_INFO(grid_cell))->tissue_type
#define HU_LAYER_BORDER_ZONE(grid_cell) (HU_FIBROTIC_INFO(grid_cell))->layer_border_zone

#define HU_INITIALIZE_FIBROTIC_INFO(grid_cell)                                                                                                                 \
    do {                                                                                                                                                       \
        size_t __size__ = sizeof(struct fibrotic_and_layered_border_zone_mesh_info);                                                                           \
        (grid_cell)->mesh_extra_info = malloc(__size__);                                                                                                       \
        (grid_cell)->mesh_extra_info_size = __size__;                                                                                                          \
        HU_FIBROTIC((grid_cell)) = false;                                                                                                                      \
        HU_BORDER_ZONE(grid_cell) = false;                                                                                                                     \
        HU_TISSUE_TYPE((grid_cell)) = 0;                                                                                                                       \
        HU_LAYER_BORDER_ZONE((grid_cell)) = 0;                                                                                                                 \
    } while(0)

#endif /* __CUSTOM_MESH_INFO_DATA_H */