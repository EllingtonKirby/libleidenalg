#ifndef CONTIGUOUSCONSTRAINEDVERTEXPARTITION_H
#define CONTIGUOUSCONSTRAINEDVERTEXPARTITION_H

#include "LinearResolutionParameterVertexPartition.h"

class LIBLEIDENALG_EXPORT ContiguousConstrainedVertexPartition : public LinearResolutionParameterVertexPartition
{
  public:
    ContiguousConstrainedVertexPartition(Graph* graph,
          vector<size_t> const& membership, double resolution_parameter);
    ContiguousConstrainedVertexPartition(Graph* graph,
          vector<size_t> const& membership);
    ContiguousConstrainedVertexPartition(Graph* graph,
      double resolution_parameter);
    ContiguousConstrainedVertexPartition(Graph* graph);
    virtual ~ContiguousConstrainedVertexPartition();
    virtual ContiguousConstrainedVertexPartition* create(Graph* graph);
    virtual ContiguousConstrainedVertexPartition* create(Graph* graph, vector<size_t> const& membership);

    virtual double diff_move(size_t v, size_t new_comm);
    virtual double quality(double resolution_parameter);

  protected:
  private:
  inline double compute_online_variance(vector<double> const& feature_weight, vector<double> const& squared_feature_weight, size_t num_features, size_t size) {
    double total_variance = 0.0;
    for (size_t f = 0; f < num_features; f++) {
      double feature = feature_weight[f];
      double x_i = squared_feature_weight[f];
      double mu_k = (feature * feature) / size;
      total_variance += x_i - mu_k;
    }

    return total_variance;
  }
};

#endif // CONTIGUOUSCONSTRAINEDVERTEXPARTITION_H
