# LJ-potential simulation
## Overview
The program is inspire by the website [https://www.complexfluids.ethz.ch/mdgpu/doc/html/group__LJforce.html](https://www.complexfluids.ethz.ch/mdgpu/doc/html/index.html). It will perform the simulation once the user input initial temperature(in K) and number of simulation steps. The program will output a file named 'particle.xyz'. It can perfrom visualized simulation in the program VMD<br/>
The simulation used parameter from argon particle and the initial velocity is initialized by Boltzmann distribution.
## Components of the program
### __device__ void ljforce( FLOAT_ARRAY_TYPE* acce,FLOAT_ARRAY_TYPE* posa, FLOAT_ARRAY_TYPE* posb,double *boxsize,double *cutoffsqr)
The function is to calculate the force of a particle to another particle and store the result in acce. If the seperation of any pair of particles is larger than cutoff distance, it will reutrn to 0.
### __device__ void gpuDistPBC( FLOAT_ARRAY_TYPE * dDist, FLOAT_ARRAY_TYPE * dA, FLOAT_ARRAY_TYPE * dB, double* boxSize)
Calculate the x,y,z direction seperation of particle and return the result in dDist. The distance will consider the periodic boundary condition(PBC).
### __device__ inline int gpuRnd(double b)
A value check whether the PBC need to be performed.
### __global__ void gpuLJForces(FLOAT_ARRAY_TYPE* dPos, FLOAT_ARRAY_TYPE* dForce, int N, double boxsize,double cutoff)
To calculation force on a particle due to all other particle.
### __device__ inline void gpuCumulateVec(FLOAT_ARRAY_TYPE* dVec, FLOAT_ARRAY_TYPE* dVecToAdd)
To add the vector dVectoAdd to dVec
### __global__ void gpuResetForces(FLOAT_ARRAY_TYPE* dForces, int N)
Set all force to 0 for all particle.
### __global__ void gpuIntegratorLeapFrog(FLOAT_ARRAY_TYPE * dPos, FLOAT_ARRAY_TYPE * dVel, FLOAT_ARRAY_TYPE * dForce, int N, double dt, double boxsize)
Perform Leap frog scheme for one time step. It will also check for PBC
### __host__ __device__ inline void checkPBC(FLOAT_ARRAY_TYPE * pos, double* boxSize)
If the particle is outside the box, it will perform the PBC on that particle.
### __global__ void gpuComponentSumReduce1(FLOAT_ARRAY_TYPE* dInData, FLOAT_ARRAY_TYPE* dInterData, int numInterData, int numThreads, int N)
Add all value dInData into dInterData in one block in GPU. The dIndata will be represented as array for each result in each block.
### __global__ void gpuComponentSumReduce2(FLOAT_ARRAY_TYPE* dInterData, int numInterData, int numThreads)
Add all value in dInterData and store the result into first element in dInterData[0]
### void componentSumReduction(FLOAT_ARRAY_TYPE* dInData, FLOAT_ARRAY_TYPE* dInterData, int numInterData, int numThreads, int N)
Add all value from dInterData to dInData. The result will be stored into dIndata[0]. It will run gpuComponentSumReduce1, then gpuComponentSumReduce2 to enhance the performance of the program.
### __global__ void gpuRemoveDriftKernel(int N, FLOAT_ARRAY_TYPE* dVel, FLOAT_ARRAY_TYPE* compSum)
Perform centre of mass correction for each vector. The result will stored in dVel.
### void removeDrift(int N, int numInterData, int numThreads, int threads, int blocks, FLOAT_ARRAY_TYPE * dVel, FLOAT_ARRAY_TYPE * compSum)
To perform the whole precodure for centre of mass correction.
###  void write(int i, FLOAT_ARRAY_TYPE* pos)
Write the current position into file "particle.xyz"
