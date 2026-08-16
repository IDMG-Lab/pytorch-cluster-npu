
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_FARTHEST_POINT_SAMPLING_H_
#define ACLNN_FARTHEST_POINT_SAMPLING_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnFarthestPointSamplingGetWorkspaceSize
 * parameters :
 * src : required
 * ptr : required
 * paddedPointPtr : required
 * outPtr : required
 * paddedOutPtr : required
 * start : required
 * mode : required
 * dist : required
 * localMaxVal : required
 * localMaxIdx : required
 * chosenLocal : required
 * syncWorkspace : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnFarthestPointSamplingGetWorkspaceSize(
    const aclTensor *src,
    const aclTensor *ptr,
    const aclTensor *paddedPointPtr,
    const aclTensor *outPtr,
    const aclTensor *paddedOutPtr,
    const aclTensor *start,
    const aclTensor *mode,
    const aclTensor *dist,
    const aclTensor *localMaxVal,
    const aclTensor *localMaxIdx,
    const aclTensor *chosenLocal,
    const aclTensor *syncWorkspace,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnFarthestPointSampling
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnFarthestPointSampling(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
