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

#include "cuda_task_class.h"
#include "cuda_tasks.h"

CUDAtask *createCUDATask(CUDAtaskNames t, int deviceID)
{
  CUDAtask *task = NULL;
  switch (t)
  {
  case Dummy:
    task = new dummyTask();
    break;
  case matrixMul:
    task = new matrixMulTask();
    break;
  case VA:
    task = new vectorAddTask();
    break;
  case BS:
    task = new BlackScholesTask();
    break;
  case PF:
    task = new PathFinderTask();
    break;
  };

  task->setDeviceID(deviceID);

  return (task);
}

void setupCUDATask(CUDAtask *task, int n, int b)
{

  if (task->getName() == Dummy)
  {
    // Dummy task parameters
    dummyTask *vt = dynamic_cast<dummyTask *>(task);
    vt->setCB(false);
    vt->setMB(true);
    if (n != 1024)
      vt->setNumElements(1024 * n);
    if (b == 0)
      vt->setPersistentBlocks();
    else if (b > 0)
    {
      dim3 blocks(b, 1, 1);
      vt->setBlocksPerGrid(blocks);
    }
    vt->setNumIterations(100); // CB 6000  MB 12000
  }
  else if (task->getName() == matrixMul)
  {
    matrixMulTask *vt = dynamic_cast<matrixMulTask *>(task);
    dim3 dA, dB;
    if (n != 1024)
    {
      dA.x = n;
      dA.y = n;
      dB.x = n;
      dB.y = n;
      vt->setMatrixDims(dA, dB);
    }
    if (b == 0)
      vt->setPersistentBlocks();
    else if (b > 0)
    {
      dim3 blocks(b, 1, 1);
      vt->setBlocksPerGrid(blocks);
    }
  }
  else if (task->getName() == VA)
  {
    vectorAddTask *vt = dynamic_cast<vectorAddTask *>(task);
    if (n != 1024)
      vt->setNumElements(n * 1024);
    if (b == 0)
      vt->setPersistentBlocks();
    else if (b > 0)
    {
      dim3 blocks(b, 1, 1);
      vt->setBlocksPerGrid(blocks);
    }
    vt->setNumIterations(100); // CB 6000  MB 12000
  }
  else if (task->getName() == BS)
  {
    BlackScholesTask *vb = dynamic_cast<BlackScholesTask *>(task);
    if (n != 1024)
      vb->setOptions(n, 256);
    if (b == 0)
      vb->setPersistentBlocks();
    else if (b > 0)
    {
      dim3 blocks(b, 1, 1);
      vb->setBlocksPerGrid(blocks);
    }
  }
  else if (task->getName() == PF)
  {
    PathFinderTask *vt = dynamic_cast<PathFinderTask *>(task);
    if (n != 1024)
      vt->setParameters(512, n, 126);
    if (b == 0)
      vt->setPersistentBlocks();
    else if (b > 0)
    {
      dim3 blocks(b, 1, 1);
      vt->setBlocksPerGrid(blocks);
    }
  }
}

int getTaskSize(CUDAtask *task)
{
  int size = 0;
  if (task->getName() == Dummy)
  {
    dummyTask *vt = dynamic_cast<dummyTask *>(task);
    size += vt->getSizeA() + vt->getSizeB() + 2 * 257 * 1024;
  }
  else if (task->getName() == matrixMul)
  {
    matrixMulTask *vtm = dynamic_cast<matrixMulTask *>(task);
    size += vtm->getSizeA() + vtm->getSizeB() + vtm->getSizeC() + 3 * 257 * 1024;
  }
  else if (task->getName() == VA)
  {
    vectorAddTask *vt = dynamic_cast<vectorAddTask *>(task);
    size += 3 * vt->getSizeN() + 3 * 257 * 1024;
  }
  else if (task->getName() == BS)
  {
    BlackScholesTask *vb = dynamic_cast<BlackScholesTask *>(task);
    size += 6 * vb->getSizeN() + 6 * 257 * 1024;
  }
  else if (task->getName() == PF)
  {
    PathFinderTask *vt = dynamic_cast<PathFinderTask *>(task);
    size += 2 * vt->getResultSize() + 2 * vt->getDataSize() + 4 * 257 * 1024;
  }
  return (size);
}

void sendTaskData(CUDAtask *task, float *d_ptr, MemoryProfiler *memprof, int c)
{
  // printf("%d reads assignments\n", getpid());
  memprof->readAssignments();
  int nA, nB, nC, nD, nE;
  int *assA = 0, *assB = 0, *assC = 0, *assD = 0, *assE = 0;

  if (task->getName() == Dummy)
  {
    dummyTask *vt = dynamic_cast<dummyTask *>(task);
    int *tmp;
    tmp = vt->getModulesA(c);
    assA = memprof->getAssignmentsIdx(vt->getSizeA(), tmp, &nA);
    tmp = vt->getModulesB(c);
    assB = memprof->getAssignmentsIdx(vt->getSizeA(), tmp, &nB);
    vt->allocDeviceMemory(d_ptr, assA, nA, assB, nB);
    vt->memHostToDeviceAssigns();
  }
  else if (task->getName() == matrixMul)
  {
    matrixMulTask *vt = dynamic_cast<matrixMulTask *>(task);
    int *tmp;
    tmp = vt->getModulesA(c);
    assA = memprof->getAssignmentsIdx(vt->getSizeA(), tmp, &nA);
    tmp = vt->getModulesB(c);
    assB = memprof->getAssignmentsIdx(vt->getSizeB(), tmp, &nB);
    tmp = vt->getModulesC(c);
    assC = memprof->getAssignmentsIdx(vt->getSizeC(), tmp, &nC);
    vt->allocDeviceMemory(d_ptr, assA, nA, assB, nB, assC, nC);
    vt->memHostToDeviceAssigns();
  }
  else if (task->getName() == VA)
  {
    vectorAddTask *vt = dynamic_cast<vectorAddTask *>(task);
    int *tmp;
    tmp = vt->getModulesA(c);
    assA = memprof->getAssignmentsIdx(vt->getSizeN(), tmp, &nA);
    tmp = vt->getModulesB(c);
    assB = memprof->getAssignmentsIdx(vt->getSizeN(), tmp, &nB);
    tmp = vt->getModulesC(c);
    assC = memprof->getAssignmentsIdx(vt->getSizeN(), tmp, &nC);
    vt->allocDeviceMemory(d_ptr, assA, nA, assB, nB, assC, nC);
    vt->memHostToDeviceAssigns();
  }
  else if (task->getName() == BS)
  {
    BlackScholesTask *vb = dynamic_cast<BlackScholesTask *>(task);
    int *tmp;
    tmp = vb->getModulesA(c);
    assA = memprof->getAssignmentsIdx(vb->getSizeN(), tmp, &nA);
    tmp = vb->getModulesB(c);
    assB = memprof->getAssignmentsIdx(vb->getSizeN(), tmp, &nB);
    tmp = vb->getModulesC(c);
    assC = memprof->getAssignmentsIdx(vb->getSizeN(), tmp, &nC);
    tmp = vb->getModulesD(c);
    assD = memprof->getAssignmentsIdx(vb->getSizeN(), tmp, &nD);
    tmp = vb->getModulesE(c);
    assE = memprof->getAssignmentsIdx(vb->getSizeN(), tmp, &nE);
    vb->allocDeviceMemory(d_ptr, assA, nA, assB, nB, assC, nC, assD, nD, assE, nE);
    vb->memHostToDeviceAssigns();
  }
  else if (task->getName() == PF)
  {
    PathFinderTask *vt = dynamic_cast<PathFinderTask *>(task);
    int *tmp;
    tmp = vt->getModulesA(c);
    assA = memprof->getAssignmentsIdx(vt->getDataSize() - vt->getResultSize(), tmp, &nA);
    tmp = vt->getModulesB(c);
    assB = memprof->getAssignmentsIdx(vt->getResultSize(), tmp, &nB);
    tmp = vt->getModulesC(c);
    assC = memprof->getAssignmentsIdx(vt->getResultSize(), tmp, &nC);
    vt->allocDeviceMemory(d_ptr, assA, nA, assB, nB, assC, nC);
    vt->memHostToDeviceAssigns();
  }
}