
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_VOXEL_GRID_H_
#define ACLNN_VOXEL_GRID_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnVoxelGridGetWorkspaceSize
 * parameters :
 * pos : required
 * size : required
 * startOptional : optional
 * endOptional : optional
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnVoxelGridGetWorkspaceSize(
    const aclTensor *pos,
    const aclTensor *size,
    const aclTensor *startOptional,
    const aclTensor *endOptional,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnVoxelGrid
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnVoxelGrid(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
