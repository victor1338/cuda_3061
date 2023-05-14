
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cuda.h>
#include <fstream>
#include <string>
#include <device_functions.h>
using namespace std;
#include <stdio.h>
#include <vector>
#include <cassert>
#include <algorithm>
#ifndef DEFINE_H
#define DEFINE_H
#include <builtin_types.h>
#include <chrono>
using namespace std::chrono;
#define FLOAT_ARRAY_TYPE float3
#endif //DEFINE_H
double const pi = 3.14f;
double const kb = 8.61f * powf(10, -5); //
double const epss = 1.03e-2f;
double const sigg = 3.40f;
double const min_r = 1.12f * sigg;

__constant__ double eps = 1.03e-2f;
__constant__ double sig = 3.40f;

__host__ __device__ inline void checkPBC(FLOAT_ARRAY_TYPE* pos, double* boxSize);

__device__ void gpuDistPBC(FLOAT_ARRAY_TYPE* dDist, FLOAT_ARRAY_TYPE* dA, FLOAT_ARRAY_TYPE* dB, double* boxSize);

__device__ inline int gpuRnd(double b);

__device__ inline void gpuCumulateVec(FLOAT_ARRAY_TYPE* dVec, FLOAT_ARRAY_TYPE* dVecToAdd);

class particle {
    double Tem;
    double bound = 5;
    double Mass;
    FLOAT_ARRAY_TYPE momen;
 

   

    double random_vel() {
        double const a = sqrt(kb * Tem / Mass);
        double q = ((double)rand() / (double)RAND_MAX);
        double w = ((double)rand() / (double)RAND_MAX);
        q = a * sqrt(-2 * log(q)) * cos(2 * pi * w);
        return q;

    };
public:
    FLOAT_ARRAY_TYPE pos[1000];
    FLOAT_ARRAY_TYPE vel[1000];


    double boxsize;
    particle() {}
    particle(double T, double M) :Tem(T), Mass(M) {
        momen.x = 0;
        momen.y = 0;
        momen.z = 0;
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 10; j++) {
                for (int k = 0; k < 10; k++) {
                    pos[i+j*10+k*100].x = bound + i*min_r; pos[i + j * 10 + k * 100].y = bound + j*min_r; pos[i + j * 10 + k * 100].z = bound + k*min_r;
                    vel[i + j * 10 + k * 100].x = random_vel(); vel[i + j * 10 + k * 100].y = random_vel(); vel[i + j * 10 + k * 100].z = random_vel();
                }
            }
        }
        boxsize = bound * 2 + 10*min_r;
        for (int i = 0; i < 1000; i++) {
            momen.x += Mass * vel[i].x; momen.y += Mass * vel[i].y; momen.z += Mass * vel[i].z;

        }
        for (int i = 0; i < 1000; i++) {
            vel[i].x -= momen.x / 1000 / Mass; vel[i].y -= momen.y / 1000 / Mass; vel[i].z -= momen.z / 1000 / Mass;

        }
    }




};

__device__ void ljforce( FLOAT_ARRAY_TYPE *acce,FLOAT_ARRAY_TYPE* posa, FLOAT_ARRAY_TYPE* posb,double *boxsize,double *cutoffsqr) {
    double r2, r6, fd;
    FLOAT_ARRAY_TYPE d;
    gpuDistPBC(&d, posa, posb, boxsize);
    r2  = (d.x * d.x + d.y * d.y + d.z * d.z);
    if((*cutoffsqr)>=r2){
        r2 = 1.0f / (d.x * d.x + d.y * d.y + d.z * d.z);
        r6 = powf(r2, 3.0f);
        fd = 48.0f * eps * r2 * r6 * (r6*powf(sig,12.0f) - 0.5f*pow(sig,6.0f));
        (*acce).x += fd * d.x;
        (*acce).y += fd * d.y;
        (*acce).z += fd * d.z;
    }
}

 __device__ void gpuDistPBC(FLOAT_ARRAY_TYPE * dDist, FLOAT_ARRAY_TYPE * dA, FLOAT_ARRAY_TYPE * dB, double* boxSize)
{
  (*dDist).x = (*dA).x - (*dB).x;
  (*dDist).y = (*dA).y - (*dB).y;
 (*dDist).z = (*dA).z - (*dB).z;
  (*dDist).x = ((*dDist).x - gpuRnd((*dDist).x / (*boxSize)) * (*boxSize));
  (*dDist).y = ((*dDist).y - gpuRnd((*dDist).y / (*boxSize)) * (*boxSize));
    (*dDist).z = ((*dDist).z - gpuRnd((*dDist).z / (*boxSize)) * (*boxSize));
 }

 __device__ inline int gpuRnd(double b)
 {
     return b < 0.0f ? static_cast<int>(b - 0.5f) : static_cast<int>(b + 0.5f);
 }

 __global__ void gpuLJForces(FLOAT_ARRAY_TYPE* dPos, FLOAT_ARRAY_TYPE* dForce, int N, double boxsize,double cutoff)
{
   int tid = threadIdx.x;
   int idx = blockDim.x * blockIdx.x + tid;
    if (idx < N)
    {
          FLOAT_ARRAY_TYPE tPos = dPos[idx];
          FLOAT_ARRAY_TYPE f;
          f.x = 0.0f;
          f.y = 0.0f;
          f.z = 0.0f;
            for (int i = 0; i < N; ++i)
              {
                  if (i != idx) 
                  {
                         ljforce(&f, &tPos, &dPos[i], &boxsize,&cutoff);
                         }
                  }
            gpuCumulateVec(&dForce[idx], &f);
      }

}

 __device__ inline void gpuCumulateVec(FLOAT_ARRAY_TYPE* dVec, FLOAT_ARRAY_TYPE* dVecToAdd)
 {
   (*dVec).x += (*dVecToAdd).x;
   (*dVec).y += (*dVecToAdd).y;
   (*dVec).z += (*dVecToAdd).z;
}

 __global__ void gpuResetForces(FLOAT_ARRAY_TYPE* dForces, int N)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
   if (idx < N)
   {
       dForces[idx].x = 0.;
        dForces[idx].y = 0.;
        dForces[idx].z = 0.;
 }
 }

__global__ void gpuIntegratorLeapFrog(FLOAT_ARRAY_TYPE * dPos, FLOAT_ARRAY_TYPE * dVel, FLOAT_ARRAY_TYPE * dForce, int N, double dt, double boxsize)
{
        int tid = threadIdx.x;
        int idx = blockDim.x * blockIdx.x + tid;
  if (idx < N)
        {
         double dx, dy, dz;

                dVel[idx].x = dVel[idx].x + dt * dForce[idx].x;
                dVel[idx].y = dVel[idx].y + dt * dForce[idx].y;
                dVel[idx].z = dVel[idx].z + dt * dForce[idx].z;

                  dx = dt * dVel[idx].x;
                  dy = dt * dVel[idx].y;
                  dz = dt * dVel[idx].z;
                  dPos[idx].x = dPos[idx].x + dx;
                 dPos[idx].y = dPos[idx].y + dy;
                dPos[idx].z = dPos[idx].z + dz;


          
                checkPBC(&dPos[idx], &boxsize);
         }
    }

__host__ __device__ inline void checkPBC(FLOAT_ARRAY_TYPE * pos, double* boxSize)
{
   (*pos).x = (*pos).x - int((*pos).x / (*boxSize)) * (*boxSize);
     if ((*pos).x < 0.0f) (*pos).x = (*pos).x + (*boxSize);
 
         (*pos).y = (*pos).y - int((*pos).y / (*boxSize)) * (*boxSize);
      if ((*pos).y < 0.0f) (*pos).y = (*pos).y + (*boxSize);
 
        (*pos).z = (*pos).z - int((*pos).z / (*boxSize)) * (*boxSize);
       if ((*pos).z < 0.0f) (*pos).z = (*pos).z + (*boxSize);
  }

__global__ void gpuComponentSumReduce1(FLOAT_ARRAY_TYPE* dInData, FLOAT_ARRAY_TYPE* dInterData, int numInterData, int numThreads, int N)
 {
 extern __shared__ FLOAT_ARRAY_TYPE smema[];
     int tid = threadIdx.x;
     int bid = blockIdx.x;
     int idx = bid * blockDim.x + tid;
     FLOAT_ARRAY_TYPE s;
     s.x = 0.0f;
     s.y = 0.0f;
     s.z = 0.0f;
     for (int j = idx; j < N; j += numInterData * numThreads)
     {
         s.x += dInData[j].x;
         s.y += dInData[j].y;
         s.z += dInData[j].z;
        }
     smema[tid].x = s.x;
     smema[tid].y = s.y;
     smema[tid].z = s.z;
     __syncthreads();
     if (tid == 0)
    {
       s.x = 0.0f;
       s.y = 0.0f;
       s.z = 0.0f;
       for (int i = 0; i < numThreads; i++)
         {
             s.x += smema[i].x;
             s.y += smema[i].y;
             s.z += smema[i].z;
         }
        dInterData[bid].x = s.x;
        dInterData[bid].y = s.y;
        dInterData[bid].z = s.z;
     }
 }

__global__ void gpuComponentSumReduce2(FLOAT_ARRAY_TYPE* dInterData, int numInterData, int numThreads)
 {
     extern __shared__ FLOAT_ARRAY_TYPE smema[];
     int tid = threadIdx.x;
     FLOAT_ARRAY_TYPE s;
     s.x = 0.0f;
     s.y = 0.0f;
     s.z = 0.0f;

     for (int j = tid; j < numInterData; j += numThreads)
     {
         s.x += dInterData[j].x;
         s.y += dInterData[j].y;
         s.z += dInterData[j].z;
     }

     smema[tid].x = s.x;
     smema[tid].y = s.y;
     smema[tid].z = s.z;
     __syncthreads();
     if (tid == 0)
     {
        s.x = 0.0f;
        s.y = 0.0f;
        s.z = 0.0f;
        for (int i = 0; i < numThreads; i++)
        {
            s.x += smema[i].x;
             s.y += smema[i].y;
             s.z += smema[i].z;
         }
        dInterData[0].x = s.x;
         dInterData[0].y = s.y;
         dInterData[0].z = s.z;
     }
 }

void componentSumReduction(FLOAT_ARRAY_TYPE* dInData, FLOAT_ARRAY_TYPE* dInterData, int numInterData, int numThreads, int N)
 {
         int smem = numThreads * sizeof(FLOAT_ARRAY_TYPE);
         gpuComponentSumReduce1 << <numInterData, numThreads, smem >> > (dInData, dInterData, numInterData, numThreads, N);
         gpuComponentSumReduce2 << <1, numThreads, smem >> > (dInterData, numInterData, numThreads);
 }

__global__ void gpuRemoveDriftKernel(int N, FLOAT_ARRAY_TYPE* dVel, FLOAT_ARRAY_TYPE* compSum)
 {
     int id = blockIdx.x * blockDim.x + threadIdx.x;
     FLOAT_ARRAY_TYPE factor;
     factor.x = -compSum[0].x / N;
     factor.y = -compSum[0].y / N;
     factor.z = -compSum[0].z / N;
     if (id < N)
     {
         dVel[id].x += factor.x;
         dVel[id].y += factor.y;
         dVel[id].z += factor.z;
     }
 }

void removeDrift(int N, int numInterData, int numThreads, int threads, int blocks, FLOAT_ARRAY_TYPE * dVel, FLOAT_ARRAY_TYPE * compSum)
{
       //get component wise sum of positon
        componentSumReduction(dVel, compSum, numInterData, numThreads, N);
 
   //remove drift
   gpuRemoveDriftKernel << <blocks, threads >> > (N, dVel, compSum);
   }

 void write(int i, FLOAT_ARRAY_TYPE* pos) {
     ofstream of;
     fstream f;
     of.open("particle.xyz",ios::app);
     if (!of) {
         cout << "No such file found";
     }
     else {
         of << 1000 << "\nframe " << i << "\n";
         for (int j = 0; j < 1000; j++) {
             of << "Particle" << " " << pos[j].x << " " << pos[j].y << " " << pos[j].z << "\n";
         }
     }
     of.close();
 }

int main()
{
    srand((unsigned)time(NULL));
    double cutoff = 2.5 * sigg;
    double cutoffsqr = cutoff * cutoff;
    int N = 1000;
    int steps;
    int redM = 128;
    int redN = 128;
    double T, M=1.0f;
    cout << "input Temperature " << endl;
    cin >> T ;
    cout << "Number of steps" << endl;
    cin >> steps;
    double dt = 0.01;
    particle atom(T,M);
    FLOAT_ARRAY_TYPE *d_pos;
    FLOAT_ARRAY_TYPE *d_velocity;
    FLOAT_ARRAY_TYPE *d_acceleration;
    FLOAT_ARRAY_TYPE* compReduction;
    int NUM_THREADS = 512;
    int NUM_BLOCKS = N / NUM_THREADS + (N % NUM_THREADS == 0 ? 0 : 1);
    double bound = atom.boxsize;
    ofstream Outttfile("particle.xyz", ofstream::trunc);
    Outttfile.close();
    FLOAT_ARRAY_TYPE* h_pos;
    FLOAT_ARRAY_TYPE* h_velocity;
    h_pos = new FLOAT_ARRAY_TYPE[N];
    h_velocity = new FLOAT_ARRAY_TYPE[N];
    for (int i = 0; i < 1000; i++) {
        h_pos[i].x = atom.pos[i].x; h_pos[i].y = atom.pos[i].y; h_pos[i].z = atom.pos[i].z;
        h_velocity[i].x = atom.vel[i].x; h_velocity[i].y = atom.vel[i].y; h_velocity[i].z = atom.vel[i].z;
    }

    dim3 dimGrid(NUM_THREADS);
    dim3 dimBlock(NUM_BLOCKS);

    cudaError_t cudaStatus;
    cudaStatus=cudaMalloc((void**)&d_pos, 1000 * sizeof(FLOAT_ARRAY_TYPE));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!(position)");
        goto Error;
    }

    cudaMalloc((void**)&compReduction, redM * sizeof(float3));

    cudaStatus = cudaMalloc((void**)&d_velocity, 1000 * sizeof(FLOAT_ARRAY_TYPE));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!(velocity)");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&d_acceleration, 1000 * sizeof(FLOAT_ARRAY_TYPE));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!(acceleration)");
        goto Error;
    }

    cudaStatus = cudaMemcpy(d_pos, h_pos, 1000 * sizeof(FLOAT_ARRAY_TYPE), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(d_velocity, h_velocity, 1000 * sizeof(FLOAT_ARRAY_TYPE), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    auto start = high_resolution_clock::now();
    gpuResetForces <<<dimGrid, dimBlock >>> (d_acceleration, 1000);
    cout << "[0/" << steps << "]";
    for (int i = 0; i < steps; i++) {
        removeDrift(N, redM, redN,NUM_THREADS, NUM_BLOCKS, d_velocity, compReduction);
        gpuResetForces << <dimGrid, dimBlock >> > (d_acceleration, 1000);
        gpuLJForces << <dimGrid, dimBlock >> > (d_pos,d_acceleration,1000,bound,cutoffsqr);
        gpuIntegratorLeapFrog << <dimGrid, dimBlock >> > (d_pos,d_velocity, d_acceleration, 1000, dt,bound);
        cudaMemcpy(h_pos, d_pos, sizeof(FLOAT_ARRAY_TYPE) * N, cudaMemcpyDeviceToHost);
        write(i, h_pos);
        cout << "\33[2K\r";
        cout << "[" << i + 1 << "/" << steps << "]";

    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(stop - start);
    cout << endl;
    cout << "Used " << duration.count()<<"seconds" << endl;
    cin >> T;





    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    cudaFree(d_pos);
    cudaFree(d_velocity);
    cudaFree(d_acceleration);

    return 0;

Error:
    cudaFree(d_pos);
    cudaFree(d_velocity);
    cudaFree(d_acceleration);

    return 0;

}
