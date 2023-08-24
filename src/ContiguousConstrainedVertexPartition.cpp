#include "ContiguousConstrainedVertexPartition.h"

#ifdef DEBUG
#include <iostream>
using std::cerr;
using std::endl;
#endif

ContiguousConstrainedVertexPartition::ContiguousConstrainedVertexPartition(Graph* graph,
      vector<size_t> const& membership, double resolution_parameter, double disconnect_penalty) :
        LinearResolutionParameterVertexPartition(graph,
        membership, resolution_parameter)
{ 
  this->disconnect_penalty = disconnect_penalty;
}

ContiguousConstrainedVertexPartition::ContiguousConstrainedVertexPartition(Graph* graph,
      vector<size_t> const& membership) :
        LinearResolutionParameterVertexPartition(graph,
        membership)
{ 
}

ContiguousConstrainedVertexPartition::ContiguousConstrainedVertexPartition(Graph* graph,
      double resolution_parameter, double disconnect_penalty) :
        LinearResolutionParameterVertexPartition(graph, resolution_parameter)
{ 
  this->disconnect_penalty = disconnect_penalty;
}

ContiguousConstrainedVertexPartition::ContiguousConstrainedVertexPartition(Graph* graph) :
        LinearResolutionParameterVertexPartition(graph)
{ 
}

ContiguousConstrainedVertexPartition::~ContiguousConstrainedVertexPartition()
{ }

ContiguousConstrainedVertexPartition* ContiguousConstrainedVertexPartition::create(Graph* graph)
{
  return new ContiguousConstrainedVertexPartition(graph, this->resolution_parameter, this->disconnect_penalty);
}

ContiguousConstrainedVertexPartition* ContiguousConstrainedVertexPartition::create(Graph* graph, vector<size_t> const& membership)
{
  return new ContiguousConstrainedVertexPartition(graph, membership, this->resolution_parameter, this->disconnect_penalty);
}

/*****************************************************************************
  Returns the difference in cluster variance if we move a node to a new community
*****************************************************************************/
double ContiguousConstrainedVertexPartition::diff_move(size_t v, size_t new_comm)
{
  #ifdef DEBUG
    cerr << "double ContiguousConstrainedVertexPartition::diff_move(" << v << ", " << new_comm << ")" << endl;
  #endif
  size_t old_comm = this->_membership[v];
  vector<double> vertex_feature_weight = this->graph->node_feature_weight(v);
  double num_active_communities = this->n_communities();
  size_t num_features = this->graph->n_node_features();
  #ifdef DEBUG
    for (size_t f = 0; f < num_features; f++) 
    {
      cerr << "Node feature weight[" << f << "]: " << vertex_feature_weight[f] << endl;
    }
    cerr << "Target comm: " << new_comm << ", " << "Old comm: " << old_comm << endl;
  #endif
  double diff = 0.0;
  if (new_comm != old_comm)
  {
    // Extract relevant features for both current and target communities
    vector<double> feature_weight_in_old_comm = this->feature_weight_in_comm(old_comm);
    vector<double> squared_feature_weight_in_old_comm = this->squared_feature_weight_in_comm(old_comm);
    
    vector<double> feature_weight_in_new_comm = this->feature_weight_in_comm(new_comm);
    vector<double> squared_feature_weight_in_new_comm = this->squared_feature_weight_in_comm(new_comm);

    size_t node_size = this->graph->node_size(v);
    size_t old_community_size = this->csize(old_comm);
    size_t new_community_size = this->csize(new_comm);
    #ifdef DEBUG
      for (size_t f = 0; f < num_features; f++) 
      {
        cerr << "Before move: Examining feature " << f << endl;
        cerr << "Before move: Current comm feature weight: " << feature_weight_in_old_comm[f] << ". Current comm squared feature weight: " << squared_feature_weight_in_old_comm[f] << endl;
        cerr << "Before move: Target comm feature weight:  " << feature_weight_in_new_comm[f] << ". Target comm squared feature weight:  " << squared_feature_weight_in_new_comm[f] << endl;
        cerr << "Before move: Current comm community size: " << old_community_size << endl;
        cerr << "Before move: Target comm community size:  " << new_community_size << endl;
      }
    #endif


    // Calculate the current feature weights
    double current_community_weight = this->resolution_parameter * num_active_communities;

    double target_community_current_variance = 0.0;
    if (new_community_size > 0) 
    {
      target_community_current_variance = this->compute_online_variance(feature_weight_in_new_comm, squared_feature_weight_in_new_comm, num_features, new_community_size);
    }
    double current_community_current_variance = this->compute_online_variance(feature_weight_in_old_comm, squared_feature_weight_in_old_comm, num_features, old_community_size);

    #ifdef DEBUG
      cerr << "Before move: Current comm Variance:   " << current_community_current_variance << endl;
      cerr << "Before move: Target comm variance:    " << target_community_current_variance << endl;
      cerr << "Before move: Num active communitites: " << num_active_communities << endl;
      cerr << "Before move: Community Weight:        " << current_community_weight << endl;
    #endif

    // Now calculate the new features if we perform the move
    for (size_t f = 0; f < num_features; f++){
      double feature_weight = vertex_feature_weight[f];
      feature_weight_in_old_comm[f] -= feature_weight;
      squared_feature_weight_in_old_comm[f] -= feature_weight * feature_weight;
      
      feature_weight_in_new_comm[f] += feature_weight;
      squared_feature_weight_in_new_comm[f] += feature_weight * feature_weight;
    }

    new_community_size += node_size;
    old_community_size -= node_size;
    #ifdef DEBUG
      for (size_t f = 0; f < num_features; f++) 
      {
        cerr << "After move: Examining feature " << f << endl;
        cerr << "After move: Current comm feature weight: " << feature_weight_in_old_comm[f] << ". Current comm squared feature weight: " << squared_feature_weight_in_old_comm[f] << endl;
        cerr << "After move: Target comm feature weight:  " << feature_weight_in_new_comm[f] << ". Target comm squared feature weight:  " << squared_feature_weight_in_new_comm[f] << endl;
        cerr << "After move: Current comm community size: " << old_community_size << endl;
        cerr << "After move: Target comm community size:  " << new_community_size << endl;
      }
    #endif

    // Adjust community sizes after move
    if (new_community_size == node_size) 
    {
      // We have created a new community
      num_active_communities += 1;
    }
    if(old_community_size == 0)
    {
      // We have eliminated a community
      num_active_communities -= 1;
    }
    double new_community_weight = this->resolution_parameter * num_active_communities;

    // Recalculate variances
    double target_community_new_variance = this->compute_online_variance(feature_weight_in_new_comm, squared_feature_weight_in_new_comm, num_features, new_community_size);
    double current_community_new_variance = 0.0;
    if(old_community_size > 0) 
    {
      current_community_new_variance = this->compute_online_variance(feature_weight_in_old_comm, squared_feature_weight_in_old_comm, num_features, old_community_size);
    }

    #ifdef DEBUG
      cerr << "After move: Current comm Variance: " << current_community_new_variance << endl;
      cerr << "After move: Target comm variance:  " << target_community_new_variance << endl;
      cerr << "After move: Num active communitites: " << num_active_communities << endl;
      cerr << "After move: Community Weight:      " << new_community_weight << endl;
    #endif

    if (!check_community_connected_except_target(old_comm, v))
    {
      // std::cout << "Found that " << old_comm << " is disconnected if we move " << v << " to " << new_comm << endl;
      return this->disconnect_penalty;
    }

    // Calculate deltas in variance and community weight if we apply this move
    double current_community_variance_delta = current_community_new_variance - current_community_current_variance;
    double target_community_variance_delta = target_community_new_variance - target_community_current_variance;
    double community_weight_delta = new_community_weight - current_community_weight;
    
    // Sum all deltas as diff in objective function
    diff = current_community_variance_delta + target_community_variance_delta + community_weight_delta;

    #ifdef DEBUG
      cerr << "Current community variance delta: " << current_community_variance_delta << endl;
      cerr << "Target community variance delta:  " << target_community_variance_delta << endl;
      cerr << "Community weight delta:           " << community_weight_delta << endl;
      cerr << "diff: " << diff << endl;
    #endif
  }
  #ifdef DEBUG
    cerr << "exit ContiguousConstrainedVertexPartition::diff_move(" << v << ", " << new_comm << ")" << endl;
    cerr << "return " << -diff << endl << endl;
  #endif
  return -diff;
}

/*****************************************************************************
  Give the variance of the partition.
******************************************************************************/
double ContiguousConstrainedVertexPartition::quality(double resolution_parameter)
{
  #ifdef DEBUG
    cerr << "double ContiguousConstrainedVertexPartition::quality()" << endl;
  #endif
  #ifdef DEBUG
    cerr << "double ContiguousConstrainedVertexPartition::resolution_parameter: " << this->resolution_parameter << endl;
  #endif
  double mod = 0.0;
  double n_communities = this->n_communities();
  size_t num_features = this->graph->n_node_features();
  size_t num_active_communities = 0;
  for (size_t c = 0; c < this->n_communities(); c++)
  {
    size_t community_size = this->csize(c);
    vector<double> feature_weight = this->feature_weight_in_comm(c);
    vector<double> squared_feature_weight = this->squared_feature_weight_in_comm(c);
    double online_variance = 0.0;
    if (community_size > 0) 
    {
      num_active_communities++;
      online_variance = this->compute_online_variance(feature_weight, squared_feature_weight, num_features, community_size);
    }
    #ifdef DEBUG
      double csize = this->csize(c);
      cerr << "\t" << "Comm: " << c << ", size: " << community_size << ", online_variance=" << online_variance;
      for (size_t f = 0; f < num_features; f++) {
        cerr << ", feature[" << f << "]=" << feature_weight[f] << ", squared weight=" << squared_feature_weight[f];
      }
      cerr << endl;
    #endif
    mod += online_variance;
  }
  #ifdef DEBUG
    cerr << "Num active communitites: " << num_active_communities << endl;
  #endif
  double q = mod + (this->resolution_parameter * num_active_communities);
  #ifdef DEBUG
    cerr << "exit double ContiguousConstrainedVertexPartition::quality()" << endl;
    cerr << "return " << -q << endl << endl;
  #endif
  return -q;
}

bool ContiguousConstrainedVertexPartition::check_community_connected_except_target(size_t v_comm, size_t target) 
{
  vector<bool> seen_nodes = vector<bool>(this->get_graph()->vcount(), false);
  vector<size_t> community = this->get_community(v_comm);

  if (community.size() < 2)
  {
    return true;
  }

  size_t index = 0;
  for (size_t i = 0; i < community.size(); i++)
  {
    if (community[i] != target)
    {
      index = i;
    }
  }

  deque<size_t> vertex_order = deque<size_t>();
  vertex_order.push_back(community[index]);
  int num_seen_elements = 0;

  while (!vertex_order.empty()) 
  {
    size_t v = vertex_order.front(); vertex_order.pop_front();
    if (v == target)
    {
      continue;
    }
    if (seen_nodes[v]) 
    {
      continue;
    }
    vector<size_t> neighbors = this->get_graph()->get_neighbours(v, IGRAPH_ALL);
    for (size_t neighbor : neighbors)
    {
      if (this->membership(neighbor) == v_comm)
      {
        vertex_order.push_back(neighbor);
      }
    }

    seen_nodes[v] = true;
    num_seen_elements++;
  }
  
  return num_seen_elements == this->csize(v_comm) - 1;
}
