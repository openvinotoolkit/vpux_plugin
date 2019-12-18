#ifndef OPERATION_PRECEDENCE_DAG_HPP
#define OPERATION_PRECEDENCE_DAG_HPP

#include <cassert>
#include <unordered_map>

#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/iterator/control_context.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "scheduler/feasible_scheduler.hpp"

namespace mv {

template<typename Model>
struct model_traits {
  typedef int const_operation_iterator_t;
  typedef int const_child_operation_iterator_t;
  typedef Model model_t;

  static const_operation_iterator_t begin_operations(model_t&);
  static const_child_operation_iterator_t
      begin_child_operations(const_operation_iterator_t&);
  static const_operation_iterator_t end_operations(model_t& model);
}; // struct model traits //


template<>
struct model_traits<mv::ControlModel> {
  typedef mv::ControlModel model_t;
  typedef mv::Control::OpListIterator const_operation_iterator_t;
  typedef mv::Control::OpChildIterator const_child_operation_iterator_t;

  //TODO(vamsikku): reference to model must be const here //
  static const_operation_iterator_t begin_operations(model_t& cm) {
    return cm.getFirst();
  }

  static const_child_operation_iterator_t begin_child_operations(
      const_operation_iterator_t& op) {
    return op.leftmostChild();
  }

  static const_operation_iterator_t end_operations(model_t& model) {
    return model.opEnd();
  }
}; // struct model_traits<mv::ControlModel> //

template<>
struct model_traits<mv::OpModel> {
  typedef mv::OpModel model_t;
  typedef mv::Data::OpListIterator const_operation_iterator_t;
  typedef mv::Data::OpChildIterator const_child_operation_iterator_t;


  static const_operation_iterator_t begin_operations(model_t& cm) {
    return cm.getInput();
  }

  static const_child_operation_iterator_t begin_child_operations(
      const_operation_iterator_t& itr) {
    return itr.leftmostChild();
  }

  static const_operation_iterator_t end_operations(model_t& model) {
    return model.opEnd();
  }
}; // struct model_traits<mv::OpModel> //


namespace scheduler {

template<typename Model=mv::OpModel>
class Operation_Dag {
  public:

    ////////////////////////////////////////////////////////////////////////////
    typedef Model model_t;
    typedef model_traits<model_t> mtraits;
    typedef typename mtraits::const_operation_iterator_t op_itr_t;
    typedef typename mtraits::const_child_operation_iterator_t child_op_itr_t;

    typedef Operation_Dag dag_t;
    typedef mv::Op const * operation_t; // &(base_node_class::content_) //
    typedef operation_t const * const_op_ptr_t;
    typedef std::hash<operation_t> operation_hash_t;


    typedef std::list<const_op_ptr_t> op_ref_list_t;
    typedef op_ref_list_t::const_iterator const_ref_op_iterator_t;

    typedef std::unordered_set<operation_t> ops_set_t;
    typedef typename ops_set_t::const_iterator const_master_op_iterator_t;
    typedef typename ops_set_t::iterator master_op_iterator_t;

    typedef std::unordered_map<operation_t, op_ref_list_t> adjacency_map_t;
    typedef typename adjacency_map_t::const_iterator const_adj_map_iterator_t;
    typedef typename adjacency_map_t::iterator adj_map_iterator_t;

    typedef std::unordered_map<operation_t, unsigned> resource_utility_map_t;
    typedef typename resource_utility_map_t::const_iterator
        const_resource_map_iterator_t;
    typedef typename resource_utility_map_t::iterator resource_map_iterator_t;
    typedef std::unordered_map<operation_t, op_itr_t> op_to_iterator_lookup_t;

    typedef std::unordered_map<operation_t, size_t> in_degree_map_t;
    typedef typename in_degree_map_t::const_iterator const_in_degree_iterator_t;

    typedef std::unordered_map<std::string, operation_t> op_name_table_t;

    class const_operation_iterator_t {
      public:
        const_operation_iterator_t()
          : ref_itr_(), master_itr_(), is_ref_type_() {}

        const_operation_iterator_t(const const_ref_op_iterator_t& ref_itr) 
          : ref_itr_(ref_itr), master_itr_(), is_ref_type_(true) {}

        const_operation_iterator_t(
            const const_master_op_iterator_t& master_itr) : ref_itr_(),
          master_itr_(master_itr), is_ref_type_(false) {}

        const operation_t& operator*() const {
          return is_ref_type_ ? *(*ref_itr_) : *master_itr_;
        }

        const_operation_iterator_t& operator++() {
          if (is_ref_type_) {
            ++ref_itr_;
          } else {
            ++master_itr_;
          }
          return *this;
        }

        bool operator==(const const_operation_iterator_t& o) const {
          if (is_ref_type_) {
            return ref_itr_ == o.ref_itr_;
          } else {
            return master_itr_ == o.master_itr_;
          }
        }

        bool operator!=(const const_operation_iterator_t& o) const {
          return !(*this == o);
        }

      private:

        const_ref_op_iterator_t ref_itr_;
        const_master_op_iterator_t master_itr_;
        bool is_ref_type_;
    }; // class const_operation_iterator_t //

    typedef size_t delay_t;
    typedef size_t resource_t;
    typedef mv::lp_scheduler::Producer_Consumer_Contiguous_Resource<resource_t,
            operation_t> resource_state_t;
    ////////////////////////////////////////////////////////////////////////////

    Operation_Dag(model_t& model) : adj_map_(), adj_map_rev_(),
      op_name_table_(), ops_(), resource_utility_map_(),
      op_to_iterator_lookup_(), in_degree_map_(), input_op_(),
      implicit_op_types_( {"Slice", "Crop"} ) {
        init_from_model(model);
    }

    void reset(model_t& model) { init_from_model(model); }

    template<typename OpTypeIterator>
    void set_implicit_op_types(OpTypeIterator begin, OpTypeIterator end) {
      static_assert(std::is_same<typename OpTypeIterator::value_type,
          std::string>::value, "Invalid OpTypeIterator");
      implicit_op_types_.clear();
      for (; begin != end; ++begin) { implicit_op_types_.push_back(*begin); }
    }

    const_operation_iterator_t begin_parent_nodes(const operation_t& op) const {
      typename adjacency_map_t::const_iterator itr = adj_map_rev_.find(op);

      return (itr == adj_map_rev_.end()) ?
        const_operation_iterator_t( ops_.end() ) :
        const_operation_iterator_t( (itr->second).begin() );
    }

    const_operation_iterator_t end_parent_nodes(const operation_t& op) const {
      typename adjacency_map_t::const_iterator itr = adj_map_rev_.find(op);

      return (itr == adj_map_rev_.end()) ?
        const_operation_iterator_t( ops_.end() ) :
        const_operation_iterator_t( (itr->second).end() );
    }

    const_operation_iterator_t begin_nodes() const {
      return const_operation_iterator_t( ops_.begin() );
    }
    const_operation_iterator_t end_nodes() const {
      return const_operation_iterator_t( ops_.end() );
    }

    // operations on the outgoing edges //
    const_operation_iterator_t begin_nodes(const operation_t& op) const {
      typename adjacency_map_t::const_iterator itr = adj_map_.find(op);

      return (itr == adj_map_.end()) ?
        const_operation_iterator_t( ops_.end() ) :
        const_operation_iterator_t( (itr->second).begin() );
    }

    const_operation_iterator_t end_nodes(const operation_t& op) const {
      typename adjacency_map_t::const_iterator itr = adj_map_.find(op);

      return (itr == adj_map_.end()) ?
        const_operation_iterator_t( ops_.end() ) :
        const_operation_iterator_t( (itr->second).end() );
    }

    resource_t resource_utility(const operation_t& op) const {
      auto itr = resource_utility_map_.find(op);
      assert(itr != resource_utility_map_.end());
      return itr->second; 
    }

    bool op_has_unit_out_degree(const operation_t& op) const {
      const_operation_iterator_t citr = begin_nodes(op),
                                 citr_end = end_nodes(op);
      if (citr == citr_end) { return false; }
      ++citr;
      return (citr == citr_end);
    }

    // Precondition: out degree of op >= 1 //
    operation_t get_first_child_op(const operation_t& op) const {
      const_operation_iterator_t citr = begin_nodes(op);
      return *citr;
    }


    // Checks if there is a DMATask which relocates the output of this op to DDR
    // Precondition: all implicit ops ("Slice" or "Crop") must be short
    // circuited.
    bool is_output_of_this_compute_op_relocated(const operation_t& op) const {
      if (!is_dpu_op(op) || !op_has_unit_out_degree(op)) { return false; }

      // does this have out degree 1 and connected to a DMATask which
      // moves data from CMX2DDR //

      operation_t cop = get_first_child_op(op); 
      return is_dma_op_moving_data_from_cmx_to_ddr(cop);
    }

    // Precondition: is_output_of_this_compute_op_relocated(op) = true //
    operation_t get_output_relocating_dma_op(const operation_t& op) const {
      assert(is_output_of_this_compute_op_relocated(op));
      return get_first_child_op(op);
    }

    ////////////////////////////////////////////////////////////////////////////

    static const_operation_iterator_t operations_begin(const dag_t& in) {
      return in.begin_nodes();
    }
    static const_operation_iterator_t operations_end(const dag_t& in) {
      return in.end_nodes();
    }

    static const_operation_iterator_t outgoing_operations_begin(const dag_t& in,
        const operation_t& op) {
      return in.begin_nodes(op);
    }
    static const_operation_iterator_t outgoing_operations_end(const dag_t& in,
        const operation_t& op) {
      return in.end_nodes(op);
    }



    static const_operation_iterator_t incoming_operations_begin(const dag_t& in,
        const operation_t& op) {
      return in.begin_parent_nodes(op);
    }
    static const_operation_iterator_t incoming_operations_end(const dag_t& in,
        const operation_t& op) {
      return in.end_parent_nodes(op);
    }


    static bool is_data_operation(const dag_t& dag, const operation_t& op) {
      return dag.is_dma_op(op) &&
          !(dag.is_dma_op_moving_data_from_cmx_to_ddr(op));
    }
    static bool is_compute_operation(const dag_t& dag, const operation_t& op) {
      // an implicit op is a compute op which takes 0 resources //
      return !(is_data_operation(dag, op));
    }

    static bool is_empty_demand(const resource_t& demand) {
      return (demand == resource_t(0UL));
    }



    static void initialize_resource_upper_bound(const resource_t& upper_bound,
        resource_state_t& state) {
      state.initialize_resource_upper_bound(upper_bound);
    }

    static bool is_resource_available(const resource_t& demand,
          const resource_state_t& state) {
      return state.is_resource_available(demand);
    }

    static bool schedule_operation(const operation_t& op,
        const resource_t& demand, resource_state_t& state,
        const_operation_iterator_t op_begin,
        const_operation_iterator_t op_end) {

        return (op->getOpType() == "Input") ?
          state.assign_resources(op, demand, op_end, op_end) :
          state.assign_resources(op, demand, op_begin, op_end);
    }

    static bool unschedule_operation(const operation_t& op,
        resource_state_t& state) {
      return state.unassign_resources(op);
    }

    static resource_t resource_utility(const dag_t& in, const operation_t& op) {
      return in.resource_utility(op);
    }

    static delay_t delay(const dag_t&, const operation_t&) {
      return delay_t(1UL);
    }
    ////////////////////////////////////////////////////////////////////////////

    op_itr_t get_op_iterator(operation_t op) const {
      typename op_to_iterator_lookup_t::const_iterator itr =
          op_to_iterator_lookup_.find(op);

      assert(itr != op_to_iterator_lookup_.end());
      return itr->second;
    }
    size_t operation_in_degree(operation_t op) const {
      const_in_degree_iterator_t itr = in_degree_map_.find(op);
      return (itr == in_degree_map_.end()) ? 0UL : itr->second;
    }
    
    bool is_input_op(operation_t op) const {
      return op->getOpType() == "Input";
    }

    operation_t get_input_op() const { return input_op_; }

    bool is_dma_op(operation_t op) const {
      return op->getOpType() == "DMATask";
    }

    bool is_dpu_op(operation_t op) const {
      return op->getOpType() == "DPUTask";
    }

    bool has_edge_between_ops(operation_t a, operation_t b) const {
      const_operation_iterator_t citr = begin_nodes(a), citr_end = end_nodes(a);
      for (; citr != citr_end; ++citr) {
        if (*citr == b) { return true; }
      }
      return false;
    }


    template<typename BackInsertIterator>
    size_t find_all_ops_exceeding_resource_threshold(resource_t threshold,
        BackInsertIterator output) {
      // TODO(vamsikku): the space can be improved by maintaining this table
      // for current level and previous level.
      typedef std::unordered_map<operation_t, resource_t> op_size_table_t;
      op_size_table_t op_size_table;
      std::list<operation_t> bfs_list;
      operation_t curr_op;

      // add all zero in-degree nodes //
      for (const_operation_iterator_t citr=begin_nodes(), citr_end=end_nodes();
            citr != citr_end; ++citr) {
        curr_op = *citr;
        size_t in_degree;
        if (!(in_degree=operation_in_degree(curr_op))) {
          printf("zero-degree-node=%s\n", curr_op->getName().c_str());
          bfs_list.push_back(curr_op);
        } else {
          printf("non-zero-degree-node=%s in-degree=%lu\n",
              curr_op->getName().c_str(), in_degree);
        }
      }


      while (!bfs_list.empty()) {
        curr_op = bfs_list.front();

        bfs_list.pop_front();
        op_size_table_t::iterator itr = op_size_table.find(curr_op);


        resource_t curr_op_utility = resource_utility(curr_op);

        if (itr == op_size_table.end()) {
          // initialize it with its output size //
          itr = op_size_table.insert(
                std::make_pair(curr_op, resource_t(0UL))).first;
        }
        itr->second += curr_op_utility;

        // for all the out-going edges
        const_operation_iterator_t citr=begin_nodes(curr_op);
        const_operation_iterator_t citr_end=end_nodes(curr_op);
        for (; citr != citr_end; ++citr) {
          operation_t child_op = *citr;
          itr = op_size_table.find(child_op);

          if (itr == op_size_table.end()) {
            // initialize it with its output size //
            itr = op_size_table.insert(
                std::make_pair(child_op, resource_t(0UL))).first;
            // newly discovered node //
            bfs_list.push_back(child_op);
          }
          itr->second += curr_op_utility;
        }

      } // while (!bfs_list.empty()) //

      size_t ret_value = 0UL;
      for (op_size_table_t::const_iterator itr=op_size_table.begin();
            itr != op_size_table.end(); ++itr) {
        if (itr->second >= threshold) {
          output = std::make_pair(itr->first, itr->second);
          ++output;
          ++ret_value;
        }
      }
      return ret_value;
    }

    operation_t get_op_by_name(const char *name) const {
      op_name_table_t::const_iterator itr = op_name_table_.find(name);
      return (itr != op_name_table_.end()) ? itr->second : operation_t(NULL);
    }

    bool is_spilled_op(operation_t op) const {
      return does_opname_ends_with(op, "spilledWrite") ||
        does_opname_have_substring(op, "_spilledRead");
    }
    bool ops_of_same_category(operation_t op_a, operation_t op_b) const {
      if (op_a->getOpType() != op_b->getOpType()) { return false; }

      if (op_a->getOpType() == "DMATask") {
        return (op_a->get<mv::DmaDirection>("direction")) ==
            (op_b->get<mv::DmaDirection>("direction"));
      }
      return true;
    }
    bool does_opname_have_substring(operation_t op, const char *substr) const {
      const std::string& op_name = op->getName();
      return !(op_name.find(substr) == std::string::npos);
    }

    bool does_opname_ends_with(operation_t op, const char *suffix) const {
      /*checks if the name ends with _spilledWrite*/
      const char *name = (op->getName()).c_str();
      size_t name_len = strlen(name), suffix_len = strlen(suffix);
      if (name_len < suffix_len) { return false; }
      if (!suffix_len) {  return true; }

      const char *rev_name_ptr = &name[name_len - 1UL];
      const char *rev_suffix_ptr = &suffix[suffix_len - 1UL];
      for (size_t i=0; i<suffix_len; i++, --rev_name_ptr, --rev_suffix_ptr) {
        if (*rev_name_ptr != *rev_suffix_ptr) { return false; }
      }
      return true;
    }

    bool is_implicit_op(operation_t op) const {
      return (op->getOpType() == "ImplicitConcat") || 
          (op->getOpType() == "Slice") || (op->getOpType() == "Crop");
    }


    bool short_circuit_unit_indegree_unit_outdegree_op(operation_t& op) {
      operation_t parent_op, child_op;
      {
        adjacency_map_t::const_iterator adj_rev_itr = adj_map_rev_.find(op);
        adjacency_map_t::const_iterator adj_itr = adj_map_.find(op);

        if ((adj_rev_itr == adj_map_rev_.end()) ||
              (adj_itr == adj_map_.end()) ) {
          return false;
        }

        const op_ref_list_t& parent_list = adj_rev_itr->second;
        const op_ref_list_t& child_list = adj_itr->second;

        if ((parent_list.size() != 1UL) || (child_list.size() != 1UL)) {
          return false;
        }

        parent_op = *(parent_list.front());
        child_op = *(child_list.front());
      }

      printf("[Short-Circuiting: (%s) -> (%s) -> (%s)]\n",
          parent_op->getName().c_str(), op->getName().c_str(),
          child_op->getName().c_str());
      fflush(stdout);

      // remove this op from the DAG //
      remove_op_from_dag(op);

      // add an edge between parent_op and child_op //
      return add_directed_edge(parent_op, child_op);
    }

    bool short_circuit_all_unit_indegree_outdegree_ops_of_this_type(
        const std::string& op_type) {
      std::list<operation_t> remove_list;

      for (const_operation_iterator_t itr=begin_nodes(); itr!=end_nodes();
            ++itr) {
        const operation_t& op = *itr;
        if (op->getOpType() == op_type) {
          remove_list.push_back(op);
        }
      }

      for (auto oitr=remove_list.begin(); oitr!=remove_list.end(); ++oitr) {
        bool short_circuited =
            short_circuit_unit_indegree_unit_outdegree_op(*oitr);
        assert(short_circuited);
      }
      return true;
    }

    struct implicit_op_color_functor_t {
      bool operator()(const dag_t& dag, const operation_t& op) const {
        return dag.is_implicit_op(op);
      }
    }; // struct implicit_op_color_functor_t //


    // Takes a operation precedence DAG with some colored ops (implicit)
    // and removes them from the DAG by adding more edges to retain operation
    // precedence invariant.
    void perform_implicit_op_color_closure() {
      typedef mv::lp_scheduler::Color_Connected_Vertices<dag_t>
          color_closure_t;

      color_closure_t color_closure_algo(*this);
      for (const_operation_iterator_t itr=begin_nodes(); itr!=end_nodes();
          ++itr) {

        // compute the color-closure of DMATask or DPUTask //
        operation_t pop = *itr;
        if (is_implicit_op(pop)) { continue; }

        std::list<operation_t> color_closure;
        color_closure_algo.compute_connected_vertices(pop,
            std::back_inserter(color_closure), implicit_op_color_functor_t() );

        printf("[ColorClosure(%s) : {", (pop->getName()).c_str());

        if (!color_closure.empty()) {
          for (auto citr=color_closure.begin(); citr!=color_closure.end();
                ++citr) {
            const operation_t& cop = *citr;
            add_directed_edge(pop, cop);
            printf(" %s ", (cop->getName()).c_str());
          }
        }
        printf("}\n");

      } // foreach implicit op in the input DAG //


      for (const_operation_iterator_t itr=begin_nodes(); itr!=end_nodes();) {
        operation_t pop = *itr;
        if (is_implicit_op(pop)) {
          const_operation_iterator_t itr_next = itr;
          ++itr_next;
          printf("[Removed %s]\n", ((*itr)->getName()).c_str());
          remove_op_from_dag(*itr);
          itr = itr_next;
        } else {
          ++itr;
        }
      }
    }



  private:

    bool is_operation_ignored(operation_t op) const {
      const std::string& op_type = op->getOpType();
      return (op_type == "ConstantInt") || (op_type == "ConstantDataElement");
    }

    void init_from_model(model_t& model) {
      adj_map_.clear();
      adj_map_rev_.clear();
      op_name_table_.clear();
      ops_.clear();
      resource_utility_map_.clear();
      op_to_iterator_lookup_.clear();
      in_degree_map_.clear();
      input_op_ = NULL;

      size_t num_ops=0UL;
      for (op_itr_t itr = mtraits::begin_operations(model);
            itr != mtraits::end_operations(model); ++itr) {
        operation_t op = &(*itr);
        if (is_operation_ignored(op)) { continue; }
        if (is_input_op(op)) { input_op_ = op; }

        ++num_ops;

        master_op_iterator_t pop_itr = ops_.find(op), cop_itr;

        if (pop_itr == ops_.end()) {
          pop_itr = (ops_.insert(op)).first;
          // op should have an unique name //
          const char * const op_name = op->getName().c_str();
          op_name_table_t::iterator nitr =
              op_name_table_.find(op->getName().c_str());
          assert(nitr == op_name_table_.end());
          op_name_table_.insert(std::make_pair(op_name, op));
        }
        op_to_iterator_lookup_.insert(std::make_pair(op, itr));

        adj_map_iterator_t adj_itr = adj_map_.find(op);
        assert(adj_itr == adj_map_.end());

        // create a new adjacency map entry //
        adj_itr = (adj_map_.insert(std::make_pair(op, op_ref_list_t()))).first;

        // adjacency list of the ops //
        op_ref_list_t &adj_list = adj_itr->second;
        for (child_op_itr_t citr = itr.leftmostChild();
              citr != model.opEnd(); ++citr) {
          operation_t child_op = &(*citr);
          if (is_operation_ignored(child_op)) { continue; }

          cop_itr = ops_.find(child_op);
          if (cop_itr == ops_.end()) {
            cop_itr = (ops_.insert(child_op)).first;
            const char * const child_op_name = child_op->getName().c_str();
            op_name_table_t::iterator nitr =
                op_name_table_.find(child_op->getName().c_str());
            assert(nitr == op_name_table_.end());
            op_name_table_.insert(std::make_pair(child_op_name, child_op));
          }

          if (in_degree_map_.find(child_op) == in_degree_map_.end()) {
            in_degree_map_[child_op] = 0UL;
          }
          in_degree_map_[child_op]++;

          adj_list.push_back( &(*cop_itr) );
          adj_map_rev_[child_op].push_back( &(*pop_itr) );
        }

        resource_t resource_utility; 

        if ( !does_the_op_run_on_hardware(op) ||
            is_dma_op_moving_data_from_cmx_to_ddr(op) ) {
          resource_utility = 0UL;
        } else {
          resource_utility = op->getOutputSize();
        }

        // resource utility //
        resource_utility_map_.insert(std::make_pair(op, resource_utility ));
      }

      // short circuit implicit ops //
      for (auto short_circuit_itr=implicit_op_types_.begin();
          short_circuit_itr!=implicit_op_types_.end(); ++short_circuit_itr) {
        short_circuit_all_unit_indegree_outdegree_ops_of_this_type(
            *short_circuit_itr);
      }

      printf("[Initfrom Model] op count = %lu\n", num_ops);
    }

    // Removes the op from the DAG and removes all incoming and outgoing edges
    void remove_op_from_dag(operation_t op) {
      // STEP-0: Find from op_set_ //
      master_op_iterator_t op_itr = ops_.find(op);
      assert(op_itr != ops_.end());

      // STEP-2: Remove from the indegree map //
      in_degree_map_.erase(op);

      // STEP-3: Remove this op from the adj_map_ of parent //
      {
        adjacency_map_t::iterator parent_itr = adj_map_rev_.find(op);

        if (parent_itr != adj_map_rev_.end()) {
          op_ref_list_t& parent_list = parent_itr->second;

          for(op_ref_list_t::const_iterator parent=parent_list.begin();
                parent!=parent_list.end(); ++parent) { //foreach parent //

            // find the adjacency list of this parent and remove op from it //
            adjacency_map_t::iterator parent_adj_itr =
                adj_map_.find(*(*parent));
            assert(parent_adj_itr != adj_map_.end());

            // remove op from this parent adjacency list //
            (parent_adj_itr->second).remove(&(*op_itr));
          }
        }

      }
        
      // STEP-4: Remove this op from the adj_map_rev_ of all its children //
      {
        adjacency_map_t::iterator child_itr = adj_map_.find(op);

        if (child_itr != adj_map_rev_.end()) {
          op_ref_list_t& child_list = child_itr->second;

          for(op_ref_list_t::const_iterator child=child_list.begin();
                child!=child_list.end(); ++child) { //foreach child//

            // find the rev-adjacency list of this child and remove op from it 
            adjacency_map_t::iterator child_adj_itr =
                adj_map_rev_.find(*(*child));
            assert(child_adj_itr != adj_map_rev_.end());

            // remove op from this child rev-adjacency list //
            (child_adj_itr->second).remove(&(*op_itr));

            // STEP-4.1: reduce the in-degree of the child //
            in_degree_map_t::iterator indegree_itr =
                in_degree_map_.find(*(*child));
            assert(indegree_itr != in_degree_map_.end());
            assert(indegree_itr->second >= 1);

            --(indegree_itr->second);
            if (!(indegree_itr->second)) {
              in_degree_map_.erase(indegree_itr);
            }

          }
        }
      }

      // STEP-1 //
      ops_.erase(op_itr);
      op_name_table_.erase(op->getName().c_str());
    } 

    bool add_directed_edge(operation_t source_op, operation_t sink_op) {

      master_op_iterator_t itr_source = ops_.find(source_op);
      master_op_iterator_t itr_sink = ops_.find(sink_op);

      if ((itr_source == ops_.end()) || (itr_sink == ops_.end())) {
        return false;
      }

      // add sink_op to adj_list of source_op //
      op_ref_list_t *child_list_ptr = NULL, *parent_list_ptr = NULL;
      {
        adjacency_map_t::iterator adj_itr = adj_map_.find(source_op);
        assert(adj_itr != adj_map_.end());

        op_ref_list_t& child_list = adj_itr->second;
        for (op_ref_list_t::const_iterator child=child_list.begin();
              child!=child_list.end(); ++child) {
          if (*child == &(*itr_sink)) { return false; }
        }
        child_list_ptr = &child_list;
      }

      // add source_op to rev_adj_list of sink_op //
      {
        adjacency_map_t::iterator adj_rev_itr = adj_map_rev_.find(sink_op);
        assert(adj_rev_itr != adj_map_rev_.end());

        op_ref_list_t& parent_list = adj_rev_itr->second;
        for (op_ref_list_t::const_iterator parent=parent_list.begin();
              parent!=parent_list.end(); ++parent) {
          if (*parent == &(*itr_source)) { return false; }
        }
        parent_list_ptr = &parent_list;
      }

      child_list_ptr->push_back(&(*itr_sink));
      parent_list_ptr->push_back(&(*itr_source));

      // update the indegree of sink_op //
      in_degree_map_t::iterator in_degree_itr = in_degree_map_.find(sink_op);

      if (in_degree_itr == in_degree_map_.end()) {
        in_degree_map_.insert(std::make_pair(sink_op, 1UL));
      } else {
        in_degree_itr->second++;
      }
      return true;
    }


    bool does_the_op_run_on_hardware(operation_t op) const {
      return (op->getOpType() == "DMATask") || (op->getOpType() == "DPUTask");
    }

    bool is_dma_op_moving_data_from_cmx_to_ddr(operation_t op) const {
      if ((op->getOpType()) != "DMATask") { return false; }
      
      mv::DmaDirectionEnum dma_dir = op->get<mv::DmaDirection>("direction");

      return (dma_dir == mv::DmaDirectionEnum::NNCMX2DDR) ||
          (dma_dir == mv::DmaDirectionEnum::UPACMX2DDR);
    }


    //TODO(vamsikku): consolidate ops_ and op_to_iterator_lookup_ tables. //
    adjacency_map_t adj_map_;
    adjacency_map_t adj_map_rev_;
    op_name_table_t op_name_table_;
    ops_set_t ops_;
    resource_utility_map_t resource_utility_map_;
    op_to_iterator_lookup_t op_to_iterator_lookup_;
    in_degree_map_t in_degree_map_;
    operation_t input_op_;
    std::vector<std::string> implicit_op_types_;
}; // class Operation_Dag //


typedef mv::lp_scheduler::Feasible_Schedule_Generator< Operation_Dag<> >
  mv_lp_scheduler_t;

typedef mv::lp_scheduler::Feasible_Schedule_Generator<
  Operation_Dag<mv::ControlModel> > mv_control_lp_scheduler_t;

} // namespace scheduler //
} // namespace mv //

namespace mv {
namespace lp_scheduler {

template<>
struct scheduler_traits< mv::scheduler::Operation_Dag<> >
  : public mv::scheduler::Operation_Dag<> {
  using mv::scheduler::Operation_Dag<>::Operation_Dag;
}; // scheduler_traits<mv::scheduler::Operation_Dag> //

}
}

namespace mv {
namespace lp_scheduler {

template<>
struct scheduler_traits< mv::scheduler::Operation_Dag<mv::ControlModel> >
  : public mv::scheduler::Operation_Dag<mv::ControlModel> {
  using mv::scheduler::Operation_Dag<mv::ControlModel>::Operation_Dag;
}; // scheduler_traits<mv::scheduler::Operation_Dag> //


typedef Feasible_Memory_Schedule_Generator< mv::scheduler::Operation_Dag<> >
  mv_memory_scheduler_with_spilling_t;

} // namespace lp_scheduler //
} // namespace mv //


#endif
