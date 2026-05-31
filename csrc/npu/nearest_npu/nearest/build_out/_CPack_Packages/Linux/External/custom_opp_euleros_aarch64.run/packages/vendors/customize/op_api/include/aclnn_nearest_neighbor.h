
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_NEAREST_NEIGHBOR_H_
#define ACLNN_NEAREST_NEIGHBOR_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnNearestNeighborGetWorkspaceSize
 * parameters :
 * x : required
 * y : required
 * ptrX : required
 * ptrY : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnNearestNeighborGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *y,
    const aclTensor *ptrX,
    const aclTensor *ptrY,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnNearestNeighbor
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnNearestNeighbor(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
