/**
 * @file cuda_task.cu
 * @author José María González Linares (jgl@uma.es)
 * @brief
 * @version 0.1
 * @date 2021-02-16
 *
 * @copyright Copyright (c) 2021
 *
 */

#include "cuda_tasks.h"

CUDAtask *createCUDATask(CUDAtaskNames t) {
  CUDAtask *task = NULL;
  switch (t) {
  case VA:
    task = new vectorAddTask();
    break;
  case BS:
    break;
  };

  return (task);
}