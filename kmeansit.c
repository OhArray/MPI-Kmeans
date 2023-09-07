float distance(float *, float *, int );
int closest_cluster(float *, float *, int , int);
void write_centroids(float *, int, int);
void write_cluster_map(int *, int);

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <assert.h>
#include <math.h>

int main(int argc, char **argv)
{
    int row = atoi(argv[2]);
    int dim = atoi(argv[3]); 
    int k = atoi(argv[4]);

    int rank, size;
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    start_time = MPI_Wtime();


    int row_per_rank =  row / size;
    int i, j;

    if (row_per_rank != floor(row_per_rank))
    {
        fprintf(stderr, "Rows is not divisible by num of ranks");
        exit(1);
    }

    float *sub_data = malloc(row_per_rank * dim * sizeof(float));
    float *dist_sum = malloc(k * dim * sizeof(float));
    float *centroids= malloc(k * dim * sizeof(float));
    int *cluster_element_count = malloc(k * sizeof(int));
    int *sub_cluster_map = malloc(row_per_rank * sizeof(int));
    float *data = NULL;
    float *global_dist_sum = NULL;
    int *global_cluster_element_count = NULL;
    int *global_cluster_map;

    if (rank == 0)
    {
        data = (float*)malloc(sizeof(float) * dim * row);
        int count = 0;
        FILE *fp = fopen(argv[1], "r");
        while (count < row *dim)
        {
            fscanf(fp, "%f", &data[count]);
            count = count + 1;
        }
        fclose(fp);        
        for (i = 0; i < k*dim; i = i+2)
        {
            int div = ((row) - k) / k;
            centroids[i] = data[i * div];
            centroids[i + 1] = data[(i * div) + 1];
        }
        global_dist_sum = malloc(k * dim * sizeof(float));
        global_cluster_element_count = malloc(k * sizeof(int));
        global_cluster_map = malloc(row_per_rank * size * sizeof(int));
    }

    MPI_Scatter(data, dim * row_per_rank, MPI_FLOAT, sub_data, dim * row_per_rank, MPI_FLOAT, 0, MPI_COMM_WORLD);

    int max_iteration = 100;

    int iteration = 0;

    while (iteration < max_iteration)
    { 
        MPI_Bcast(centroids, k * dim, MPI_FLOAT, 0, MPI_COMM_WORLD);

        for (i = 0; i < k * dim; i++) 
        {
            dist_sum[i] = 0.0;
            if (i < k) 
            {
                cluster_element_count[i] = 0;
            }
        }

        float *data_point = sub_data;
        for (i = 0; i < row_per_rank; i++)
        {
            int current_cluster = closest_cluster(data_point, centroids, k, dim);
            cluster_element_count[current_cluster]++;
            for (int j = 0; j < dim; j++)
            {
                dist_sum[current_cluster * dim + j] += data_point[j];
            }
            data_point = data_point + dim;
        }

        MPI_Reduce(dist_sum, global_dist_sum, k * dim, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(cluster_element_count, global_cluster_element_count, k, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0)
        {
            for (i = 0; i < k; i++)
            {
                for (j = 0; j < dim; j++)
                {
                    int dij = dim * i + j;
                    if (global_cluster_element_count[i] != 0)
                    {
                        global_dist_sum[dij] = global_dist_sum[dij] / global_cluster_element_count[i];
                    }
                }
            }
            for (i = 0; i < k * dim; i++)
            {
                centroids[i] = global_dist_sum[i];
                write_centroids(centroids, k, dim);
            }
            iteration++;
        }
        MPI_Bcast(&iteration, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    float *data_point = sub_data;
    for (i = 0; i < row_per_rank; i++)
    {
        sub_cluster_map[i] = closest_cluster(data_point, centroids, k, dim);
        data_point = data_point + dim;
    }

    MPI_Gather(sub_cluster_map, row_per_rank, MPI_INT, global_cluster_map, row_per_rank, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        write_cluster_map(global_cluster_map, row);
        end_time = MPI_Wtime();
        printf("Execution time: %f seconds\n", end_time - start_time);
    }


    MPI_Finalize();
}

float distance(float *a, float *b, int dim)
{
  int i;
  float sum = 0, res = 0;
  for (i = 0; i < dim; i++)
  {
    float diff = (a[i] - b[i]);
    sum = sum + (diff * diff);
  }
  res = sqrt(sum);
  return res;
}

int closest_cluster(float *data_point, float *centroids, int k, int dim)
{
    int i;
    int current_cluster = 0;
    float current_distance = distance(data_point, centroids, dim);
    float *centroid = centroids + dim;
    for (i = 1; i < k; i++)
    {
        float clust_distance = distance(data_point, centroid, dim);
        if (clust_distance < current_distance)
        {
            current_cluster = i;
            current_distance = clust_distance;
        }
        centroid = centroid + dim;
    }
    return current_cluster;
}

void write_centroids(float *centroids, int k, int dim)
{
    float *out = centroids;
    FILE *fa = fopen("centroids.out", "w");
    int i;
    for (i = 0; i < k; i++)
    {
        int j;
        for (j = 0; j < dim; j++, out++)
        {
            fprintf(fa, "%f ", *out);
        }
        fprintf(fa,"\n");
    }
    fclose(fa);
}

void write_cluster_map(int *global_cluster_map, int row)
{
    FILE *fb = fopen("cluster_map.out", "w");
    int i;
    for (i = 0; i < row; i++)
    {
        fprintf(fb, "%d\n", global_cluster_map[i]);
    }
    fclose(fb);
}