#pragma once

#include <algorithm>
#include <map>
#include <string>
#include <limits>
#include <torch/script.h>

#include "dsa/arch/estimation.h"
#include "dsa/arch/model.h"
#include "schedule.h"
#include "scheduler.h"

#include "dsa/core/singleton.h"

// This class contains all info which you might want to remember about
class WorkloadSchedules {
 public:
  std::vector<Schedule> sched_array;

  float estimate_performance() { return 0; }
};

static unsigned int log2 (unsigned int val) {
    if (val == 0) return -1;
    if (val == 1) return 0;
    unsigned int ret = 0;
    while (val > 1) {
        val >>= 1;
        ret++;
    }
    return ret;
};

class SchedStats {
 public:
  int64_t lat = std::numeric_limits<int64_t>::max();
  int64_t latmis = std::numeric_limits<int64_t>::max();
  int64_t agg_ovr = std::numeric_limits<int64_t>::max();
  int64_t ovr = std::numeric_limits<int64_t>::max();
  int64_t max_util = std::numeric_limits<int64_t>::max();
};

// Class which captures a codesign between hardware and software through
// scheduling.  Note here the the Model and all schedules are "owned" by
// the codesign instance, ie. they will be deleted when the class is deleted
//
// TODO: There are an infinite number of things to do here, starting with:
// 1. Explore full design space with randomness
// 2.

inline int next_power_of_two(int n) {
    if (n <= 1) return 1;
    // decrement `n` (to handle the case when `n` itself
    // is a power of 2)
    n = n - 1;
 
    // initialize result by 2
    int k = 2;
 
    // double `k` and divide `n` in half till it becomes 0
    while (n >>= 1) {
        k = k << 1;    // double `k`
    }
 
    return k;
}

template <typename T>
inline int non_uniform_random(const std::vector<T>& nodes, const std::vector<bool>& vec) {
  for (int res = rand() % nodes.size();; res = rand() % nodes.size()) {
    if (vec[nodes[res]->id()]) {
      return res;
    }
    if (rand() % 2 == 0) {
      return res;
    }
  }
}

class CodesignInstance {
  SSModel _ssModel;

 public:
  SSModel* ss_model() { return &_ssModel; }
  ~CodesignInstance() {}
  void add_workloads(const std::string &filename, bool is_list);
  std::vector<WorkloadSchedules> workload_array;
  std::vector<double> weight;

  std::vector<Schedule*> res;
  std::pair<double, double> objective{-1, -1};
  int num_cores = 1;
  int num_banks = 8;
  int system_bus = 8;

  /*
    Vector of unused nodes and links.
    The nodes and links are in the order of the original ADG.
    False means the node or link is used
    True means the node or link is unused
  */
  std::vector<bool> unused_nodes;
  std::vector<bool> unused_links;

  bool sanity_check{false};
  std::string dse_fail_reason = "";
  std::string dse_change = "none";
  std::vector<std::string> dse_changes_log;
  double performance = 0;
  double prior_score = 0;
  double links_score = 0;
  double nodes_score = 0;
  std::vector<double> workload_performances;
  std::vector<std::vector<std::pair<std::string, double>>> dfg_performances;
  std::vector<std::vector<std::pair<std::string, std::string>>> dfg_spm_performances;
  std::vector<std::vector<std::pair<std::string, std::string>>> dfg_l2_performances;
  std::vector<std::vector<std::pair<std::string, std::string>>> dfg_dram_performances;
  std::vector<double> workload_weights;
  double normalized_resources = 0;

  CodesignInstance(SSModel* model);

  void printJson(std::string filename);

  void printScheduleStats() {
    for_each_sched([&](Schedule& sched) {
      bool succeed_sched = sched.num_left() == 0;
      SchedStats s;
      sched.get_overprov(s.ovr, s.agg_ovr, s.max_util);
      sched.fixLatency(s.lat, s.latmis, sched.violation_penalty);
      int violation = sched.violation();

      std::string name = basename(sched.ssdfg()->filename);

      DSA_INFO << "Printing Schedule: " << name;
      DSA_INFO << "Schedule Took: " << sched.scheduled_seconds << " seconds";

      if (dsa::ContextFlags::Global().verbose) {
        if (succeed_sched) {
          // Also check final latency
          DSA_INFO << "latency: " << s.lat;
          DSA_INFO << "lat-mismatch-max: " << s.latmis;
          DSA_INFO << "lat-mismatch-sum: " << violation;
          DSA_INFO << "overprov-max: " << s.ovr;
          DSA_INFO << "overprov-sum: " << s.agg_ovr;
          sched.stat_printOutputLatency();
        } else {
          DSA_INFO << "Scheduling Failed";
        }
      }

      DSA_INFO << "Printing Edges:";
      sched.printEdge();

      if (dsa::ContextFlags::Global().verbose) {
        std::string spm_performance = "";
        std::string l2_performance = "";
        std::string dram_performance = "";
        DSA_INFO << "Performance: " << sched.estimated_performance(spm_performance, l2_performance, dram_performance);
        DSA_INFO << "  SPM Bandwidth: " << spm_performance;
        DSA_INFO << "  L2 Bandwidth: " << l2_performance;
        DSA_INFO << "  DRAM Bandwidth: " << dram_performance;
      }
      
      std::string config_header = name + ".dfg.h";
      std::ofstream osh(config_header);
      DSA_CHECK(osh.good());
      sched.printConfigHeader(osh, name);


      if (dsa::ContextFlags::Global().bitstream) {
        std::string config_header_bits = name + ".dfg.bits.h";
        std::ofstream oshb(config_header_bits);
        DSA_CHECK(oshb.good());
        sched.printConfigHeader(oshb, name, true);
      }

      std::string graphviz_sched = "viz/" +  name + ".gv";
      sched.printGraphviz(graphviz_sched.c_str());
      DSA_INFO << "Sched GV Generated. Print with Command:";
      DSA_INFO << "dot -Tpng -o viz/" + name + ".png " + graphviz_sched; 
    });
  }


  adg::estimation::Result EstimatePowerArea() {
    auto resource_estimation = dsa::adg::estimation::EstimatePowerAera(ss_model());
    
    resource_estimation.add_core_overhead();
    resource_estimation.scale_cores(num_cores);
    resource_estimation.add_system_bus_overhead(num_cores, num_banks, system_bus);

    return resource_estimation;
  }

  // Check that everything is okay
  /*!
   * \brief The per-node link orders must admit one global connection order.
   * dsagen2 assigns Diplomacy port indices from the order in which links are
   * connected and derives that order from the SourceIndex/SinkIndex pairs
   * (same source: higher index later; same sink: higher index later); if that
   * relation has a cycle the generator rejects the ADG. Append-only mutations
   * preserve acyclicity, positional inserts can break it.
   */
  void verify_link_order() {
    auto* sub = _ssModel.subModel();
    int n = sub->link_list().size();
    std::vector<int> indeg(n, 0);
    std::vector<std::vector<int>> succ(n);
    std::set<ssnode*> nodes(sub->node_list().begin(), sub->node_list().end());
    for (auto* data : sub->data_list()) nodes.insert(data);
    for (auto* node : nodes) {
      for (bool in : {false, true}) {
        auto& links = in ? node->in_links() : node->out_links();
        for (int i = 1; i < (int) links.size(); ++i) {
          if (!links[i - 1] || !links[i]) continue;
          succ[links[i - 1]->id()].push_back(links[i]->id());
          ++indeg[links[i]->id()];
        }
      }
    }
    std::vector<int> ready;
    for (int i = 0; i < n; ++i) if (indeg[i] == 0) ready.push_back(i);
    int visited = 0;
    while (!ready.empty()) {
      int u = ready.back(); ready.pop_back(); ++visited;
      for (int v : succ[u]) if (--indeg[v] == 0) ready.push_back(v);
    }
    if (visited != n) {
      std::ostringstream os;
      for (int i = 0; i < n; ++i) if (indeg[i] > 0) os << " " << sub->link_list()[i]->name();
      DSA_CHECK(false) << "link connection order has a cycle (dsagen2 cannot elaborate it); links involved:" << os.str();
    }
  }

  void verify() {
    verify_link_order();
    if (!sanity_check) return;
    
    DSA_CHECK(workload_array.size() > 0);
    for (auto& work : workload_array) {
      DSA_CHECK(work.sched_array.size() > 0);
    }
    
    // Check Schedules
    for_each_sched([&](Schedule& sched) {
      sched.verify();

      DSA_CHECK(&_ssModel == sched.ssModel()) << &_ssModel << " " << sched.ssModel();
      DSA_CHECK(_ssModel.subModel()->node_list().size() >= sched.node_prop().size());
      DSA_CHECK(_ssModel.subModel()->link_list().size() >= sched.link_prop().size());

      for (auto& ep : sched.edge_prop()) {
        for (auto& p : ep.links) {
          DSA_CHECK(p.second->id() < (int)_ssModel.subModel()->link_list().size());
          DSA_CHECK(p.second->id() < (int)sched.link_prop().size());
          DSA_CHECK(_ssModel.subModel()->link_list()[p.second->id()] == p.second);
        }
      }
    });
    // Check Node
    for (auto& node : _ssModel.subModel()->node_list()) {
      if (auto fu = dynamic_cast<ssfu*>(node)) {
        DSA_CHECK(!fu->fu_type().capability.empty());
      } else if (auto vport  = dynamic_cast<SyncNode*>(node)) {
        if (vport->vp_stated()) {
          DSA_CHECK(vport->out_links().size() + vport->in_links().size() > 1);
        }
      }
      //DSA_CHECK(node->parent == _ssModel.subModel());
      DSA_CHECK(node == _ssModel.subModel()->node_list()[node->id()]);
    }
    
    for (auto& node : _ssModel.subModel()->node_list()) {
      for (sslink* link : node->in_links()) {
        DSA_CHECK(link->id() < (int)_ssModel.subModel()->link_list().size());
        DSA_CHECK(_ssModel.subModel()->link_list()[link->id()] == link);
      }
      for (sslink* link : node->out_links()) {
        DSA_CHECK(link->id() < (int)_ssModel.subModel()->link_list().size());
        DSA_CHECK(_ssModel.subModel()->link_list()[link->id()] == link);
      }
    }

    for (unsigned i = 0; i < _ssModel.subModel()->link_list().size(); ++i) {
      DSA_CHECK(_ssModel.subModel()->link_list()[i]->id() == (int) i);
    }
  }

  /*
    Gets a log of the changes that happened during this iteration of the dse.
    The boolean cout relates to formating for either the log file or output.
  */
  std::string get_changes_log(bool cout = false) {
    std::ostringstream s;
    for(auto it = dse_changes_log.begin(); it != dse_changes_log.end(); ++it) {
      s << *it;
      if (cout)
        s << std::endl;
      else if (std::next(it) != dse_changes_log.end())
        s << ",";
    }
    return s.str();
  }

  std::string get_workload_weights(bool cout=false) {
    std::ostringstream s;
    for(auto it = workload_weights.begin(); it != workload_weights.end(); ++it) {
      s << *it;
      if (cout)
        s << std::endl;
      else if (std::next(it) != workload_weights.end())
        s << ",";
    }
    return s.str();
  }

  std::string get_workload_performances(bool cout=false) {
    std::ostringstream s;
    for(auto it = workload_performances.begin(); it != workload_performances.end(); ++it) {
      s << *it;
      if (cout)
        s << std::endl;
      else if (std::next(it) != workload_performances.end())
        s << ",";
    }
    return s.str();
  }

  std::string get_dfg_performances(bool cout=false) {
    std::ostringstream s;
    for(auto it = dfg_performances.begin(); it != dfg_performances.end(); ++it) {
      for(auto next = it->begin(); next != it->end(); ++next) {
        s << next->first << ": " << next->second;
        if (cout)
          s << std::endl;
        else if (std::next(it) != dfg_performances.end() || std::next(next) != it->end())
          s << ",";
      }
    }
    return s.str();
  }

  std::string get_spm_performances(bool cout=false) {
    std::ostringstream s;
    for(auto it = dfg_spm_performances.begin(); it != dfg_spm_performances.end(); ++it) {
      for(auto next = it->begin(); next != it->end(); ++next) {
        s << next->first << ": " << next->second;
        if (cout)
          s << std::endl;
        else if (std::next(it) != dfg_spm_performances.end() || std::next(next) != it->end())
          s << ",";
      }
    }
    return s.str();
  }

  std::string get_l2_performances(bool cout=false) {
    std::ostringstream s;
    for(auto it = dfg_l2_performances.begin(); it != dfg_l2_performances.end(); ++it) {
      for(auto next = it->begin(); next != it->end(); ++next) {
        s << next->first << ": " << next->second;
        if (cout)
          s << std::endl;
        else if (std::next(it) != dfg_l2_performances.end() || std::next(next) != it->end())
          s << ",";
      }
    }
    return s.str();
  }

  std::string get_dram_performances(bool cout=false) {
    std::ostringstream s;
    for(auto it = dfg_dram_performances.begin(); it != dfg_dram_performances.end(); ++it) {
      for(auto next = it->begin(); next != it->end(); ++next) {
        s << next->first << ": " << next->second;
        if (cout)
          s << std::endl;
        else if (std::next(it) != dfg_dram_performances.end() || std::next(next) != it->end())
          s << ",";
      }
    }
    return s.str();
  }

  // Verify, plus also check:
  // 1. there are no hangers (elements with no purpose, usually after delete)
  void verify_strong() { verify(); }

  void add_edges_to_data_node(DataNode* n) {
    auto sub = _ssModel.subModel();
    if (sub->vport_list().empty()) return;

    for (auto ovp : sub->output_list()) {
      add_link(n, ovp);
    } 
    for (auto ivp : sub->input_list()) {
      if (memory_feeds_inputs(n)) add_link(n, ivp);
    }
  }

  void add_random_edges_to_node(ssnode* n, int min_in, int max_in, int min_out,
                                int max_out) {
    auto* sub = _ssModel.subModel();

    if (sub->node_list().empty()) return;

    int n_ins = rand() % (max_in - min_in) + min_in;
    for (int i = 0, j = 0; i < n_ins && j < n_ins * 10 && n->in_links().size() <= 4;
         ++i, ++j) {
      int src_node_index = rand() % sub->nodes<ssnode*>().size();
      ssnode* src = sub->node_list()[src_node_index];
      if ((dynamic_cast<SyncNode*>(src) && src->out_links().empty()) || src == n) {
        i--;
        continue;
      }

      if (src == n) continue;
      add_link(src, n);
      

    }
    int n_outs = rand() % (max_out - min_out) + min_out;
    for (int i = 0, j = 0; i < n_outs && j < n_outs * 10 && n->out_links().size() <= 4;
         ++i, ++j) {
      int dst_node_index = rand() % sub->node_list().size();
      ssnode* dst = sub->node_list()[dst_node_index];
      if ((dynamic_cast<SyncNode*>(dst) && dst->in_links().empty()) || dst == n) {
        i--;
        continue;
      }

      if (n == dst) continue;
      add_link(n, dst);
    }
  }

  template <typename T>
  bool delete_nodes(std::function<bool(T*)> f) {
    bool res = false;
    auto* sub = _ssModel.subModel();
    std::for_each(sub->nodes<T*>().begin(), sub->nodes<T*>().end(),
                  [this, &res, f](T* n) {
                    if (f(n)) {
                      this->delete_node(n);
                      res = true;
                    }
                  });
    return res;
  }

  /**
   * @brief Deletes Links and Nodes that can't be useful
   * 
   * @return true if something is deleted
   * @return false otherwise
   */
  bool delete_hangers() {
    bool deleted_something = false;

    //TODO: Convert from set once memory is added
    std::unordered_set<ssnode*> nodes_to_delete;

    auto* sub = _ssModel.subModel();

    for (ssfu* fu : sub->fu_list()) {
      if (fu->in_links().size() == 0 || fu->out_links().size() == 0) {
        nodes_to_delete.insert(fu);
        deleted_something = true;
      }
      /* This case doesn't work as it could be used as a passthru. Maybe convert to a switch instead?
      if (fu->fu_type().capability.empty()) {
        nodes_to_delete.insert(fu);
        deleted_something = true;
      }
      */
    }
    for (ssswitch* sw : sub->switch_list()) {
      if (sw->out_links().size() == 0 || sw->in_links().size() == 0) {
        nodes_to_delete.insert(sw);
        deleted_something = true;
      }
    }

    for (ssivport* ivport : sub->input_list()) {
      if (ivport->out_links().size() == 0) {
        nodes_to_delete.insert(ivport);
        deleted_something = true;
      }
    }
    for (ssovport* ovport : sub->output_list()) {
      if (ovport->in_links().size() == 0) {
        nodes_to_delete.insert(ovport);
        deleted_something = true;
      }
    }

    for (auto& node : nodes_to_delete) {
      delete_node(node);
    }

    return deleted_something;
  }

  bool dfg_exceeds_size(SSDfg* dfg, int size) {
    for (int i = 0; (long unsigned int) i < dfg->meta.size(); ++i) {
      if (dfg->meta[i].unroll > size) {
        return true;
      }
    }
    return false;
  }

  /**
   * @brief Use Prior to prunning. Will collapse all stated edges that are
   * unused
   * 
   * @return true if a stated edge is collapsed 
   * @return false if no stated edge is collapsed
   */
  bool all_stated_collapse() {
    while (delete_hangers()) {  }
    auto* sub = _ssModel.subModel();
    auto util = utilization();
    if (std::get<2>(util) == 0) {
      return false;
    }

    std::vector<ssnode*> nodes_to_delete;
    std::vector<sslink*> links_to_delete;

    for (int i = 0; i < sub->node_list().size(); ++i) {
      ssnode* n = sub->node_list()[i];
      if (auto sync = dynamic_cast<SyncNode*>(n)) {
        if (sync->vp_stated()) {
          if (!stated_mapped(sync) && !must_stay_stated(sync)) {
            stated_collapse(sync);
          }
        }
      }
    }
    while (delete_hangers()) {  }
    return false;
  }

  /*!
   * \brief Whether a link connects a memory (data) node with a vector port.
   * Streams are not mapped onto these links by the spatial scheduler, so link
   * utilization never marks them as used; but the hardware generator requires
   * every surviving vector port to keep at least one memory feeder/sink.
   * Such links are therefore never removed while both endpoints exist.
   */
  /*!
   * \brief A stated vector port reserves its first compute-side link for the
   * stream state; no DFG edge is routed on it, so link-usage bookkeeping sees
   * it as unused, yet removing it silently shrinks the port's capacity.
   */
  /*!
   * \brief Whether a vector port has to keep its stream state in hardware.
   * dsagen2 requires an input vector port fed by a memory that supports linear
   * padding to be stated (padding is driven by the stream state), and the
   * printer honours that rule. Exploration must therefore never un-state such
   * a port, otherwise the emitted ADG has one data link less per port than
   * the model the schedules were validated on, and the lane-to-link
   * assignment shifts.
   */
  static bool must_stay_stated(SyncNode* vport) {
    if (!vport->isInputPort()) return false;
    for (auto* link : vport->in_links()) {
      auto* data = dynamic_cast<DataNode*>(link->source());
      if (data && data->linearPadding()) return true;
    }
    return false;
  }

  /*!
   * \brief dsagen2 requires two output vector ports on every memory engine
   * that supports indirect streams (the index stream is produced by an output
   * port), so a design with indirect memories may never shrink to one OVP.
   */
  bool memory_supports_indirect() {
    for (auto* data : _ssModel.subModel()->data_list()) {
      if (data->indirectIndexStream() || data->indirectLength1DStream() ||
          data->indirectStride2DStream())
        return true;
    }
    return false;
  }
  int min_output_ports() { return memory_supports_indirect() ? 2 : 1; }

  static bool is_stated_reserved_link(sslink* link) {
    if (auto* ivp = dynamic_cast<ssivport*>(link->source()))
      return ivp->vp_stated() && !ivp->out_links().empty() && ivp->out_links()[0] == link;
    if (auto* ovp = dynamic_cast<ssovport*>(link->sink()))
      return ovp->vp_stated() && !ovp->in_links().empty() && ovp->in_links()[0] == link;
    return false;
  }

  static bool is_memory_port_link(sslink* link) {
    auto* src = link->source();
    auto* snk = link->sink();
    bool src_data = dynamic_cast<DataNode*>(src) != nullptr;
    bool snk_data = dynamic_cast<DataNode*>(snk) != nullptr;
    bool src_sync = dynamic_cast<SyncNode*>(src) != nullptr;
    bool snk_sync = dynamic_cast<SyncNode*>(snk) != nullptr;
    return (src_data && snk_sync) || (src_sync && snk_data);
  }

  /*!
   * \brief Connect a vector port to every memory (data) node, as the seed ADGs
   * do. The hardware generator requires each input port to have a memory
   * feeder and each output port a memory sink.
   */
  /*!
   * \brief Whether a memory engine may feed input vector ports. dsagen2's
   * register engine only returns output-port data to the CPU (REGImpl requires
   * #IVP == 0), every other engine feeds inputs.
   */
  static bool memory_feeds_inputs(DataNode* data) { return dynamic_cast<ssregister*>(data) == nullptr; }

  void connect_port_to_memory(ssivport* ivp) {
    auto* sub = _ssModel.subModel();
    for (auto* data : sub->data_list()) if (memory_feeds_inputs(data)) add_link(data, ivp);
  }
  void connect_port_to_memory(ssovport* ovp) {
    auto* sub = _ssModel.subModel();
    for (auto* data : sub->data_list()) add_link(ovp, data);
  }

  /*!
   * \brief Make sure every surviving vector port still has a memory link.
   */
  void repair_port_memory_links() {
    auto* sub = _ssModel.subModel();
    for (auto* ivp : sub->input_list()) {
      bool has_memory = false;
      for (auto* l : ivp->in_links())
        if (dynamic_cast<DataNode*>(l->source())) { has_memory = true; break; }
      if (!has_memory) connect_port_to_memory(ivp);
    }
    for (auto* ovp : sub->output_list()) {
      bool has_memory = false;
      for (auto* l : ovp->out_links())
        if (dynamic_cast<DataNode*>(l->sink())) { has_memory = true; break; }
      if (!has_memory) connect_port_to_memory(ovp);
    }
  }

  /*!
   * \brief Bridge disconnected islands of the compute fabric. The hardware
   * generator builds one reconfiguration network over all compute nodes and
   * vector ports by walking neighbour links, so every surviving node must be
   * reachable. Each secondary component is joined to the largest one through
   * a bidirectional pair of links between switches (or an FU if a component
   * has no switch); unused links are harmless to the existing schedules.
   */
  void repair_compute_connectivity() {
    auto* sub = _ssModel.subModel();
    std::vector<ssnode*> nodes;
    for (auto* n : sub->fu_list()) nodes.push_back(n);
    for (auto* n : sub->switch_list()) nodes.push_back(n);
    for (auto* n : sub->input_list()) nodes.push_back(n);
    for (auto* n : sub->output_list()) nodes.push_back(n);
    auto is_fabric = [](ssnode* y) {
      return y && (dynamic_cast<SpatialNode*>(y) != nullptr || dynamic_cast<SyncNode*>(y) != nullptr);
    };
    std::map<ssnode*, int> comp;
    int ncomp = 0;
    for (auto* start : nodes) {
      if (comp.count(start)) continue;
      std::vector<ssnode*> stack{start};
      comp[start] = ncomp;
      while (!stack.empty()) {
        auto* x = stack.back();
        stack.pop_back();
        auto visit = [&](ssnode* y) {
          if (is_fabric(y) && !comp.count(y)) { comp[y] = ncomp; stack.push_back(y); }
        };
        for (auto* l : x->in_links()) visit(l->source());
        for (auto* l : x->out_links()) visit(l->sink());
      }
      ++ncomp;
    }
    if (ncomp <= 1) return;
    std::vector<int> size(ncomp, 0);
    for (auto& kv : comp) size[kv.second]++;
    int main_c = std::max_element(size.begin(), size.end()) - size.begin();
    auto pick = [&](int c) -> ssnode* {
      ssnode* fallback = nullptr;
      for (auto& kv : comp) {
        if (kv.second != c) continue;
        if (dynamic_cast<ssswitch*>(kv.first)) return kv.first;
        if (!fallback && dynamic_cast<ssfu*>(kv.first)) fallback = kv.first;
      }
      return fallback;
    };
    ssnode* hub = pick(main_c);
    bool changed = false;
    for (int c = 0; c < ncomp; ++c) {
      if (c == main_c) continue;
      ssnode* other = pick(c);
      if (!hub || !other) continue;
      add_link(hub, other);
      add_link(other, hub);
      dse_changes_log.push_back("bridge disconnected compute component " + other->name());
      changed = true;
    }
    if (changed) for_each_sched([&](Schedule& sched) { sched.allocate_space(); });
  }

  /*!
   * \brief Give every vector port at least two compute-side neighbours.
   * The hardware generator builds its reconfiguration network as a spanning
   * tree with a small fan-out limit; a port that hangs as a single-link leaf
   * off a switch forces that switch to parent it, and a switch with more such
   * leaves than the fan-out limit makes the tree impossible. A second switch
   * neighbour (as in the seed ADGs) gives the tree builder an alternative.
   */
  void repair_port_redundancy() {
    auto* sub = _ssModel.subModel();
    if (sub->switch_list().size() < 2) return;
    auto fabric_degree = [&](ssnode* n) {
      int d = 0;
      for (auto* l : n->in_links()) if (!dynamic_cast<DataNode*>(l->source())) ++d;
      for (auto* l : n->out_links()) if (!dynamic_cast<DataNode*>(l->sink())) ++d;
      return d;
    };
    auto pick_switch = [&](ssnode* port, bool port_is_input) -> ssswitch* {
      ssswitch* best = nullptr;
      int best_degree = 1 << 30;
      for (auto* sw : sub->switch_list()) {
        bool adjacent = false;
        for (auto* l : port->out_links()) if (l->sink() == sw) adjacent = true;
        for (auto* l : port->in_links()) if (l->source() == sw) adjacent = true;
        if (adjacent) continue;
        int d = fabric_degree(sw);
        if (d < best_degree) { best_degree = d; best = sw; }
      }
      return best;
    };
    bool changed = false;
    for (auto* ivp : sub->input_list()) {
      if (fabric_degree(ivp) >= 2) continue;
      if (auto* sw = pick_switch(ivp, true)) { add_link(ivp, sw); changed = true; }
    }
    for (auto* ovp : sub->output_list()) {
      if (fabric_degree(ovp) >= 2) continue;
      if (auto* sw = pick_switch(ovp, false)) { add_link(sw, ovp); changed = true; }
    }
    // dsagen2 requires an input-controlled PE to have at least two inputs
    // (PEFuncUnitInst: "You support input control, but you only have 1
    // inputs"); pruning can leave a PE with a single input link.
    for (auto* fu : sub->fu_list()) {
      if (!fu->inputCtrl() || fu->in_links().size() >= 2) continue;
      if (auto* sw = pick_switch(fu, true)) { add_link(sw, fu); changed = true; }
    }
    if (changed) {
      dse_changes_log.push_back("add second switch neighbour to leaf vector ports");
      for_each_sched([&](Schedule& sched) { sched.allocate_space(); });
    }
  }

  /*!
   * \brief Newly created switches/FUs start from the model defaults (8-bit
   * granularity, 15-deep buffers), which do not match the fabric they join and
   * make the hardware generator fail on mismatched datapath bundles. Copy the
   * datapath parameters from an existing fabric node instead.
   */
  void inherit_fabric_parameters(ssnode* node) {
    auto* sub = _ssModel.subModel();
    ssnode* tmpl = nullptr;
    for (auto* sw : sub->switch_list()) if (sw != node) { tmpl = sw; break; }
    if (!tmpl) for (auto* fu : sub->fu_list()) if (fu != node) { tmpl = fu; break; }
    if (!tmpl) return;
    node->datawidth(tmpl->datawidth());
    node->granularity(tmpl->granularity());
    node->flow_control(tmpl->flow_control());
    node->max_delay(tmpl->max_delay());
  }
  ssswitch* add_switch_like_fabric() {
    ssswitch* sw = _ssModel.subModel()->add_switch();
    inherit_fabric_parameters(sw);
    return sw;
  }
  ssfu* add_fu_like_fabric() {
    ssfu* fu = _ssModel.subModel()->add_fu();
    inherit_fabric_parameters(fu);
    return fu;
  }

  void prune_all_unused() {
    auto* sub = _ssModel.subModel();

    // First Prune all current Hangers
   // while (delete_hangers()) {  }

    // Now get the utilization
    auto util = utilization();
    
    // If there is no utilization, then we shouldn't be pruning
    if (std::get<2>(util) == 0) {
      return;
    }

    // Uncache objective
    objective = std::make_pair<double, double>(-1, -1);

    std::vector<ssnode*> nodes_to_delete;
    std::vector<sslink*> links_to_delete;

    // Links and nodes that are unused by every schedule but must survive the
    // prune anyway, because the hardware needs them.
    std::set<sslink*> keep_links;
    std::set<ssnode*> keep_nodes;
    auto is_data = [](ssnode* n) { return dynamic_cast<DataNode*>(n) != nullptr; };
    auto link_survives = [&](sslink* l) {
      return !unused_links[l->id()] || is_memory_port_link(l) || is_stated_reserved_link(l) || keep_links.count(l);
    };
    auto node_survives = [&](ssnode* n) {
      return !unused_nodes[n->id()] || is_data(n) || keep_nodes.count(n);
    };
    // Keep an unused node alive without turning it into a hanger: it needs one
    // compute-side out-link to a node that survives, recursively.
    std::vector<ssnode*> chain;
    auto keep_alive = [&](ssnode* n) {
      if (node_survives(n)) return;
      keep_nodes.insert(n);
      chain.push_back(n);
    };
    auto settle_chain = [&]() {
      while (!chain.empty()) {
        ssnode* n = chain.back(); chain.pop_back();
        if (dynamic_cast<ssovport*>(n)) continue;  // memory links are its outputs
        bool has_out = false;
        for (auto* l : n->out_links()) if (!is_data(l->sink()) && link_survives(l)) has_out = true;
        if (has_out) continue;
        sslink* best = nullptr;
        for (auto* l : n->out_links()) if (!is_data(l->sink()) && node_survives(l->sink())) { best = l; break; }
        if (!best) for (auto* l : n->out_links()) if (!is_data(l->sink())) { best = l; break; }
        if (!best) continue;
        keep_links.insert(best);
        keep_alive(best->sink());
      }
    };

    // 1. A stated port's reserved link (never pruned, see is_stated_reserved
    //    link) lands on a node that may be unused otherwise; deleting that node
    //    as a hanger would take the reserved link, and the port's schedule,
    //    with it.
    for (auto* ivp : sub->input_list()) {
      if (ivp->vp_stated() && !ivp->out_links().empty()) keep_alive(ivp->out_links()[0]->sink());
    }
    settle_chain();

    // 2. Indirect-capable memories need two output vector ports in hardware
    //    (see min_output_ports); if the schedules use fewer, retain one unused
    //    OVP together with one compute-side link from a node that survives.
    if (min_output_ports() > 1) {
      int used_ovps = 0;
      for (auto* ovp : sub->output_list()) used_ovps += !unused_nodes[ovp->id()];
      if (used_ovps < min_output_ports()) {
        ssovport* spare_ovp = nullptr;
        sslink* spare_link = nullptr;
        for (auto* ovp : sub->output_list()) {
          if (!unused_nodes[ovp->id()]) continue;
          for (auto* l : ovp->in_links()) {
            if (!is_data(l->source()) && node_survives(l->source())) { spare_ovp = ovp; spare_link = l; break; }
          }
          if (spare_ovp) break;
        }
        if (!spare_ovp) {
          for (auto* ovp : sub->output_list()) if (unused_nodes[ovp->id()]) { spare_ovp = ovp; break; }
        }
        if (spare_ovp) {
          keep_nodes.insert(spare_ovp);
          if (spare_link) keep_links.insert(spare_link);
          dse_changes_log.push_back("retain " + spare_ovp->name() + " for indirect streams");
        }
      }
    }

    // Add all links that are unused to a vector then delete them
    // Note: Links must be deleted before nodes
    for (int i = 0; i < unused_links.size(); i++) {
      if (!link_survives(sub->link_list()[i]))
        links_to_delete.push_back(sub->link_list()[i]);
    }

    for (auto link : links_to_delete) {
      delete_link(link, false);
    }

    // Add all nodes that are unused to a vector then delete them.
    // The DMA and scratchpad engines are the accelerator's memory interface
    // and must survive even when no schedule marks them as used (DFGs without
    // array annotations carry no memory vertices at all); without them the
    // generated hardware cannot stream any data.
    for (int i = 0; i < unused_nodes.size(); i++) {
      ssnode* n = sub->node_list()[i];
      // Memory engines are never pruned: besides DMA/SPM, the ISA needs the
      // register engine for ss_recv (scalar read-back), the generate engine
      // for constant streams and the recurrence engine for ss_rec, none of
      // which is visible in the extracted DFGs.
      if (unused_nodes[i] && !keep_nodes.count(n) && !dynamic_cast<DataNode*>(n))
        nodes_to_delete.push_back(n);
    }

    for (auto node : nodes_to_delete) {
      delete_node(node, false, false, false);
    }

    // A retained OVP must not be a hanger: give it a compute-side link if it
    // has none (repair_port_redundancy adds the second one).
    for (auto* n : keep_nodes) {
      if (auto* ovp = dynamic_cast<ssovport*>(n)) {
        bool has_compute_link = false;
        for (auto* l : ovp->in_links()) if (!is_data(l->source())) has_compute_link = true;
        if (!has_compute_link && !sub->switch_list().empty()) add_link(sub->switch_list()[0], ovp);
      }
    }
    
    while (delete_hangers()) {  }
    repair_port_memory_links();
    repair_compute_connectivity();
    ensure_output_ports();
    repair_port_redundancy();
    return;
  }

  /*!
   * \brief Exploration can lose output ports through hanger deletion, so the
   * two-OVP rule for indirect memories (min_output_ports) is enforced on the
   * final design: missing ports are created next to an existing one.
   */
  static void copy_port_kind(ssivport* vport, ssivport* tmpl) {
    vport->repeatIVP(tmpl->repeatIVP());
    vport->broadcastIVP(tmpl->broadcastIVP());
  }
  static void copy_port_kind(ssovport* vport, ssovport* tmpl) {
    vport->discardOVP(tmpl->discardOVP());
    vport->taskOVP(tmpl->taskOVP());
  }
  // A port created during exploration must look like the ports of the seed:
  // the model's defaults (crossbar implementation, no repeat/broadcast, 16-deep
  // FIFO) describe hardware the bitstream encoder never configures, and the
  // compiler relies on repeat ports for scalars reused across a stream.
  template <typename VP>
  static void clone_port_attributes(VP* vport, const std::vector<VP*>& ports) {
    VP* tmpl = nullptr;
    for (VP* p : ports) if (p != vport) { tmpl = p; break; }
    if (!tmpl) return;
    vport->datawidth(tmpl->datawidth());
    vport->granularity(tmpl->granularity());
    vport->max_util(tmpl->max_util());
    vport->flow_control(tmpl->flow_control());
    vport->max_delay(tmpl->max_delay());
    vport->vp_impl(tmpl->vp_impl());
    vport->vp_stated(tmpl->vp_stated());
    copy_port_kind(vport, tmpl);
  }

  void ensure_output_ports() {
    auto* sub = _ssModel.subModel();
    while ((int) sub->output_list().size() < min_output_ports()) {
      ssovport* ovp = sub->add_output_vport();
      clone_port_attributes(ovp, sub->output_list());
      ovp->vp_stated(false);
      connect_port_to_memory(ovp);
      ssnode* src = nullptr;
      if (!sub->switch_list().empty()) src = sub->switch_list()[0];
      else if (!sub->fu_list().empty()) src = sub->fu_list()[0];
      if (src) add_link(src, ovp);
      for_each_sched([&](Schedule& sched) { sched.allocate_space(); });
      dse_changes_log.push_back("add " + ovp->name() + " for indirect streams");
    }
  }

  void make_random_modification(double temperature) {
    int iteration = rand() % 3;
    int add_iterations, remove_iterations, modify_iterations = 0;
    for (int i = 0; i < temperature; ++i) {
      int item_class = rand() % 100; // choose random number between 0 and 99
      bool change = false;
      switch (iteration) {
        case 0: {
          change = add_something(item_class);
          add_iterations++;
          break;
        }
        case 1: {
          change = remove_something(item_class);
          remove_iterations++;
          break;
        }
        case 2: {
          change = modify_something(item_class);
          modify_iterations++;
          break;
        }
      }
    }

    for_each_sched([&](Schedule& sched) {
      sched.verify();
    });

    if (add_iterations > remove_iterations && add_iterations > modify_iterations) {
      dse_change = "add";
      if (rand() % 100 <= temperature * temperature)
        _ssModel.io_ports += rand() % (4 - _ssModel.io_ports + 1);
    } else if (remove_iterations > add_iterations && remove_iterations > modify_iterations) {
      dse_change = "remove";
      if (rand() % 100 <= temperature * temperature)
        _ssModel.io_ports = rand() % _ssModel.io_ports + 1;
    } else {
      dse_change = "change";
      if (rand() % 100 <= temperature)
        _ssModel.io_ports = rand() % 4 + 1;
    }
  }

  /**
   * @brief Adds some hardware to the ADG
   * 
   * 65% chance to add a random link
   * 15% chance to add a random switch
   * 10% chance to add a random FU
   * 5% chance to add a random input vport
   * 5% chance to add a random output vport
   * 
   * @param item_class A random number with the item to modified
   * @return bool whether something was succesfully added in this call.
   */
  bool add_something(int item_class) {
    std::ostringstream s;
    auto* sub = _ssModel.subModel();
    if (item_class < 50) { // 50% to add a random link
      if (sub->node_list().empty()) return false;
      int src_node_index = rand() % sub->node_list().size();
      int dst_node_index = rand() % sub->node_list().size();
      ssnode* src = sub->node_list()[src_node_index];
      ssnode* dst = sub->node_list()[dst_node_index];
      if (dynamic_cast<SyncNode*>(src) && src->out_links().empty()) {
        return false;
      }
      if (dynamic_cast<SyncNode*>(dst) && dst->in_links().empty()) {
        return false;
      }
      if (src == dst) return false;

      add_link(src, dst);

      s << "add link from " << src->name() << " to " << dst->name();
    } else if (item_class < 65) { // 10% to add a random switch
      ssswitch* sw = add_switch_like_fabric();
      add_random_edges_to_node(sw, 1, 5, 1, 5);
      s << "add switch " <<  sw->name() << " ins/outs: " << sw->in_links().size() << "/" << sw->out_links().size();
    } else if (item_class < 75) { // 10% to add a random FU
      // Randomly pick an FU type from the set
      auto& fu_defs = _ssModel.fu_types;
      if (fu_defs.empty()) return false;
      ssfu* fu = add_fu_like_fabric();
      int fu_def_index = rand() % fu_defs.size();
      Capability* def = fu_defs[fu_def_index];
      fu->fu_type(*def);
      s << "add function unit " << fu->name();
    } else if (item_class < 85) { //  10% to add a random input vector port
      ssivport* vport = sub->add_input_vport();
      clone_port_attributes(vport, sub->input_list());
      connect_port_to_memory(vport);
      add_random_edges_to_node(vport, 0, 1, 5, 12);
      s << "adding input vport " << vport->name();
    } else if (item_class < 95) { // 10% to add a random output vector port
      ssovport* vport = sub->add_output_vport();
      clone_port_attributes(vport, sub->output_list());
      connect_port_to_memory(vport);
      add_random_edges_to_node(vport, 5, 12, 0, 1);
      s << "adding output vport " << vport->name();
    } else { // 5% to add a scratchpad
      // Disabled: a new scratchpad gets the model's default parameters, which
      // the hardware generator rejects next to the seed's memory nodes (e.g. a
      // constant-stream data type on a node that does not support generated
      // streams). Re-enable once new memory nodes clone an existing one.
      return false;
      ssscratchpad* spm = sub->add_scratchpad();
      add_edges_to_data_node(spm);
      s << "adding scratchpad " << spm->name();
    } 

    dse_changes_log.push_back(s.str());
    for_each_sched([&](Schedule& sched) { sched.allocate_space(); });
    while (delete_hangers()) {  }
    verify_strong();
    return true;
  }


  /**
   * @brief Removes something from the ADG
   * 
   * 60% Chance to remove a random link
   * 15% Chance to remove a random switch
   * 15% Chance to remove a random FU
   * 5% Chance to remove a random input vector port
   * 5% Chance to remove a random output vector port
   * 
   * @param item_class A random number with the item to remove
   * @return bool whether something was succesfully removed in this call.
   */
  bool remove_something(int item_class) {
    std::ostringstream s;
    auto* sub = _ssModel.subModel();
    if (item_class < 50) { // 50% to remove a random link
      if (sub->link_list().empty()) return false;
      int index = non_uniform_random(sub->link_list(), unused_links);
      sslink* l = sub->link_list()[index];
      if (is_memory_port_link(l)) return false;  // structural, see is_memory_port_link
      if (is_stated_reserved_link(l)) return false;  // reserved for stream state
      if (auto* ivp = dynamic_cast<ssivport*>(l->source())) {
        // hardware needs one state link plus at least one data link
        if (must_stay_stated(ivp) && ivp->out_links().size() <= 2) return false;
      }
      s << "remove link " << l->name();
      dse_changes_log.push_back(s.str());
      delete_link(l);
    } else if (item_class < 65) { // 15% to remove a random switch
      if (sub->switch_list().empty()) return false;
      int index = non_uniform_random(sub->switch_list(), unused_nodes);
      ssswitch* sw = sub->switch_list()[index];
      s << "remove switch " << sw->name();
      dse_changes_log.push_back(s.str());
      delete_node(sw, true, false, true);
    } else if (item_class < 80) { // 10% to remove a random FU
      if (sub->fu_list().empty()) return false;
      int index = non_uniform_random(sub->fu_list(), unused_nodes);
      ssfu* fu = sub->fu_list()[index];
      s << "remove function unit " << fu->name();
      dse_changes_log.push_back(s.str());
      delete_node(fu, true, false, false);
    } else if (item_class < 85) { // 5% to remove a random input vector port
      if (sub->input_list().size() <= 1) return false;
      int index = non_uniform_random(sub->input_list(), unused_nodes);
      ssivport* vport = sub->input_list()[index];
      s << "remove input vport "<< vport->name();
      dse_changes_log.push_back(s.str());
      delete_node(vport, false, false, false);
    } else if (item_class < 90) { // 5% to remove a random output vector port
      if ((int) sub->output_list().size() <= min_output_ports()) return false;
      int index = non_uniform_random(sub->output_list(), unused_nodes);
      ssovport* vport = sub->output_list()[index];
      s << "remove output vport "<< vport->name();
      dse_changes_log.push_back(s.str());
      delete_node(vport, false, false, false);
    } else { // 10% to remove a data node
      if (sub->data_list().empty()) return false;
      int index = non_uniform_random(sub->data_list(), unused_nodes);
      DataNode* mem = sub->data_list()[index];
      // Memory engines are the accelerator's interface to the ISA (DMA/SPM
      // for streams, register engine for ss_recv, generate engine for
      // constant streams, recurrence engine for ss_rec); the DFGs do not show
      // which ones the kernels need, so none of them is ever removed.
      return false;
      s << "remove memory " << mem->name();
      dse_changes_log.push_back(s.str());
      delete_node(mem, false, false, false);
    } 

    verify_strong();
    while (delete_hangers()) {  }
    return true;
  }

  /**
   * @brief Modifies a node in the ADG
   * 
   * 15% to change flow control
   * 5% to change the utilization of a functional unit
   * 30% chance to change the fifo depth
   * 20% chance to change a functional units fu type
   * 20% chance to change a nodes granularity
   * 5% to change input vport stated
   * 5% to change output vport stated
   * 
   * @param item_class A random number with the item to modified
   * @return bool whether something was succesfully modified in this call.
   */
  bool modify_something(int item_class) {
    std::ostringstream s;
    auto* sub = _ssModel.subModel();
    if (item_class < 15) { // 15% to change flow control
      if (sub->node_list().empty()) return false;
      int node_index = rand() % sub->node_list().size();
      ssnode* node = sub->node_list()[node_index];
      if (dynamic_cast<SyncNode*>(node)) return false;
      // Flow control is not explored: a static switch output register ignores
      // downstream ready (NBufferImpl: input.ready := true), so data is lost
      // whenever a dynamic PE or vector port stalls, and static PEs with a
      // register file elaborate into a combinational loop. The scheduler only
      // models dynamic behaviour for control-dependent nodes, so a mixed
      // static/dynamic fabric is not something it can schedule safely.
      return false;
      s << "change Node " << node->name() << " flow control from " << node->flow_control() << " to " << !node->flow_control();

      node->flow_control(!node->flow_control());
      
      if (!node->flow_control()) {
        for_each_sched([&](Schedule& sched) {
          for (int slot = 0; slot < sched.num_slots(node); ++slot) {
            for (auto& p : sched.dfg_nodes_of(slot, node)) {
              if (sched.needs_dynamic[p.first->id()]) {
                sched.unassign_dfgnode(p.first);
              }
            }
          }
        });
      }
    } else if (item_class < 20) { // 5% to change the util of a node
      return false;

      if (sub->fu_list().empty()) return false;
      // Modify FU utilization
      int diff = rand() % 16 - 8;
      if (diff == 0) return false;
      int fu_index = rand() % sub->fu_list().size();
      ssfu* fu = sub->fu_list()[fu_index];
      int old_util = fu->max_util();
      int new_util = std::max(1, old_util + diff);
      if (diff < -4) new_util = 1;

      if (old_util == 1 && new_util > 1) {
        fu->flow_control(true);
      }

      fu->max_util(new_util);
      s << "change FU " << fu->name() << " util from " << old_util << " to " << new_util;

      // if we are constraining the problem, then lets re-assign anything
      // mapped to this FU
      if (new_util < old_util) {
        unassign_node(fu);
      }

    } else if (item_class < 30) { // 10% to change fu fifo depth
      if (sub->fu_list().empty()) return false;
      int diff = -(rand() % 3 + 1);
      if (rand() & 1) diff *= -1;
      int fu_index = rand() % sub->fu_list().size();
      ssfu* fu = sub->fu_list()[fu_index];
      int old_delay_fifo_depth =  fu->delay_fifo_depth();
      // Constrain delay fifo depth between 1 and 32
      int new_delay_fifo_depth = std::min(std::max(1, old_delay_fifo_depth + diff), 32);
      fu->max_delay(new_delay_fifo_depth);

      s << "change FU " << fu->name() << " fifo depth from " << old_delay_fifo_depth << " to " << new_delay_fifo_depth;
      // if we are constraining the problem, then lets re-assign anything
      // mapped to this FU
      for_each_sched([&](Schedule& sched) {
        for (int slot = 0; slot < sched.num_slots(fu); ++slot) {
          for (auto& p : sched.dfg_nodes_of(slot, fu)) {
            for (auto& op : p.first->ops()) {
              for (auto eid : op.edges) {
                auto* elem = &sched.ssdfg()->edges[eid];
                if (sched.edge_delay(elem) > fu->delay_fifo_depth()) {
                  sched.unassign_dfgnode(p.first);
                }
              }
            }
          }
        }
      });
    } else if (item_class < 40) { // 10% to change a node's fu_type
      if (sub->fu_list().empty()) return false;
      int index = rand() % sub->fu_list().size();
      auto fu = sub->fu_list()[index];

      if (rand() & 1) {
        s << "add FU " << fu->name() << " operation";

        fu->fu_type(*ss_model()->fu_types[rand() % ss_model()->fu_types.size()]);
      } else if (fu->fu_type().capability.size() > 1) {
        int j = rand() % fu->fu_type().capability.size();
        s << "remove FU " << fu->name() << " operation " << dsa::name_of_inst(fu->fu_type().capability.at(j).op);
        auto fu_type = fu->fu_type();
        fu_type.Erase(j);
        fu->fu_type(fu_type);
        /*
        s << " new operations: \"";
        for (auto& op : fu->fu_type().capability) {
          s << dsa::name_of_inst(op.op) << " ";
        }
        s << "\"";
        */

      }

      for_each_sched([&](Schedule& sched) {
        for (int slot = 0; slot < sched.num_slots(fu); ++slot) {
          for (auto& p : sched.dfg_nodes_of(slot, fu)) {
            if (auto inst = dynamic_cast<dsa::dfg::Instruction*>(p.first)) {
              if (!fu->fu_type().Capable(inst->inst())) {
                sched.unassign_dfgnode(p.first);
              }
            }
          }
        }
      });
    } else if (item_class < 50) { // 20% to change node granularity
      return false;
      int index = rand() % sub->node_list().size();
      auto &node = sub->node_list()[index];

      if (node->datawidth() == 8) {
        return false;
      }
      
      // Doesn't make sense to change a memory or sync nodes granularity
      if (dynamic_cast<DataNode*>(node)) return false;
      if (dynamic_cast<SyncNode*>(node)) return false;

      int new_one = (1 << (rand() % 4)) * 8;
      while (new_one == node->granularity() || new_one > node->datawidth()) {
        new_one = (1 << (rand() % 4)) * 8;
        
      }

      // Unassign everything from this node
      unassign_node(node);
      for (auto link : node->in_links()) {
        unassign_link(link);
      }
      for (auto link : node->out_links()) {
        unassign_link(link);
      }
      int prev_gran = node->granularity();

      // Change the granularity
      node->granularity(new_one);


      // Redo the schedule data-structure according to new granularity
      for_each_sched([node, this, new_one](Schedule& sched) {
        auto &np = sched.node_prop()[node->id()];
        np.slots.resize(node->lanes());
        
        for (auto link : node->in_links()) {
          auto &lp = sched.link_prop()[link->id()];
          lp.slots.resize(link->lanes());
        }

        for (auto link : node->out_links()) {
          auto &lp = sched.link_prop()[link->id()];
          lp.slots.resize(link->lanes());
        }
      });
      
      node->resetAllRoutingTables();
      for (auto link : node->in_links()) {
        link->source()->resetNodeTable(node);
        if (link->source()->type() == ssnode::NodeType::InputVectorPort) {
          unassign_node(link->source());
        }
      }
      for (auto link : node->out_links()) {
        link->sink()->resetNodeTable(node);
        if (link->sink()->type() == ssnode::NodeType::OutputVectorPort) {
          unassign_node(link->sink());
        }
      }

      s << "change " << node->name() << " granularity from " << prev_gran << " to " << new_one;

    } else if (item_class < 60) { // 10% chance to change a nodes datawidth
      return false;
      int index = rand() % sub->node_list().size();
      auto node = sub->node_list()[index];

      // Doesn't make sense to change a memory or sync nodes datawidth
      if (dynamic_cast<DataNode*>(node)) return false;
      if (dynamic_cast<SyncNode*>(node)) return false;
      
      int new_one = (1 << (rand() % 5)) * 8;
      while (new_one == node->datawidth()) {
        new_one = (1 << (rand() % 4)) * 8;
      }

      if (node->granularity() > new_one) {
        return false;
      }

      // Unassign everything from this node
      unassign_node(node);
      for (auto link : node->in_links()) {
        unassign_link(link);
      }
      for (auto link : node->out_links()) {
        unassign_link(link);
      }
      int prev_data = node->datawidth();

      // Change the datawidth
      node->datawidth(new_one);


      // Redo the schedule data-structure according to new datawidth
      for_each_sched([node, this, new_one](Schedule& sched) {
        auto &np = sched.node_prop()[node->id()];
        np.slots.resize(node->lanes());
        
        for (auto link : node->in_links()) {
          auto &lp = sched.link_prop()[link->id()];
          lp.slots.resize(link->lanes());
        }

        for (auto link : node->out_links()) {
          auto &lp = sched.link_prop()[link->id()];
          lp.slots.resize(link->lanes());
        }
      });

      node->resetAllRoutingTables();
      for (auto link : node->in_links()) {
        link->source()->resetNodeTable(node);
        if (link->source()->type() == ssnode::NodeType::InputVectorPort) {
          unassign_node(link->source());
        }
      }
      for (auto link : node->out_links()) {
        link->sink()->resetNodeTable(node);
        if (link->sink()->type() == ssnode::NodeType::OutputVectorPort) {
          unassign_node(link->sink());
        }
      }

      s << "change " << node->name() << " datawidth from " << prev_data << " to " << new_one;

    } else if (item_class < 65) { // 5% to change a input Vports stated
      if  (sub->input_list().empty()) return false;
      int index = rand() % sub->input_list().size();
      auto vport = sub->input_list()[index];

      if (vport->out_links().size() < 2) return false;
      if (vport->vp_stated() && must_stay_stated(vport)) return false;

      if (vport->vp_stated()) {
        stated_collapse(vport);
      } else {
        vport->vp_stated(true);
      }
      unassign_node(vport);

      s << "change Input vport " << vport->name() << " state from " << !vport->vp_stated() << " to " << vport->vp_stated();
    }  else if (item_class < 75) { // 10% to change scratchpad capacity
      if  (sub->scratch_list().empty()) return false;
      int index = rand() % sub->scratch_list().size();
      auto spm = sub->scratch_list()[index];
      int size = (int) log2(spm->capacity());
      if (rand() & 1) {
        size += 1;
      } else {
        size -= 1;
      }
      size = std::max(size, 8);
      size = std::min(size, 21);

      spm->capacity(std::pow(2, size));
      unassign_node(spm);
      s << "change SPM Capacity " << spm->name() << " to " << spm->capacity();
    } else if (item_class < 85) { // 10% to change system bus size
      return false;
      if  (sub->dma_list().empty()) return false;
      int index = rand() % sub->dma_list().size();
      auto dma = sub->dma_list()[index];
      DSA_CHECK(dma->readWidth() == dma->writeWidth());
      int bus_width = (int) log2(dma->readWidth());
      if (rand() & 1) {
        bus_width += 1;
      } else {
        bus_width -= 1;
      }
      bus_width = std::max(bus_width, 3);
      bus_width = std::min(bus_width, 6);

      dma->readWidth(std::pow(2, bus_width));
      dma->writeWidth(std::pow(2, bus_width));
    } else if (item_class < 95) { // 10% to change scratchpad bus size
      // Disabled like the system-bus mutation above: an overlay whose
      // scratchpad bus was narrowed to 8 B computed wrong results on RTL even
      // for DMA-only kernels (memory-side vector port datapath), while the
      // same overlay with the seed's 32 B bus passed all four kernels.
      return false;
      if  (sub->scratch_list().empty()) return false;
      int index = rand() % sub->scratch_list().size();
      auto spm = sub->scratch_list()[index];
      DSA_CHECK(spm->readWidth() == spm->writeWidth());
      int bus_width = (int) log2(spm->readWidth());
      if (rand() & 1) {
        bus_width += 1;
      } else {
        bus_width -= 1;
      }
      bus_width = std::max(bus_width, 3);
      bus_width = std::min(bus_width, 5);

      spm->readWidth(std::pow(2, bus_width));
      spm->writeWidth(std::pow(2, bus_width));
      s << "change SPM width " << spm->name() << " to " << spm->readWidth();
    } else { // 5% to change a output Vports stated
      if  (sub->output_list().empty()) return false;
      int index = rand() % sub->output_list().size();
      auto vport = sub->output_list()[index];

      if (vport->in_links().size() < 2) return false;

      if (vport->vp_stated()) {
        stated_collapse(vport);
      } else {
        vport->vp_stated(true);
      }
      unassign_node(vport);

      s << "change Output vport " << vport->name() << " state from " << !vport->vp_stated() << " to " << vport->vp_stated();
    }

    if (false) {
      int index = rand() % sub->switch_list().size();
      auto sw = sub->switch_list()[index];
      if (sw->max_delay() > 0) {
        if (!check_cycle(sw))
          return false;
        // TODO: Add a check for timing constraints
        
        sw->max_delay(0);
        unassign_node(sw);
        s << "change Switch " << sw->name() << " max delay from " << sw->max_delay() << " to " << 0;
      } else {
        sw->max_delay(15);
        s << "change Switch " << sw->name() << " max delay from " << sw->max_delay() << " to " << 15;
      }
    }
    dse_changes_log.push_back(s.str());
    while (delete_hangers()) {  }
    return true;
  }


  /**
   * @brief Lambda function to loop through all schedules
   * 
   * @param f 
   */
  void for_each_sched(const std::function<void(Schedule&)>& f) {
    for (auto& ws : workload_array) {
      for (Schedule& sched : ws.sched_array) {
        f(sched);
      }
    }
  }

  /**
   * @brief Construct a new Codesign Instance object
   * 
   * @param c other codesign instance
   * @param from_scratch whether to restart from scratch
   */
  CodesignInstance(const CodesignInstance& c, bool from_scratch) : _ssModel(c._ssModel){
    objective = std::make_pair<double, double>(-1, -1);
    weight = c.weight;
    num_cores = c.num_cores;
    num_banks = c.num_banks;
    system_bus = c.system_bus;
    
    if (from_scratch) {
      for (auto& work : c.workload_array) {
        workload_array.emplace_back();
        for (auto& elem : work.sched_array) {
          workload_array.back().sched_array.emplace_back(ss_model(), elem.ssdfg());
        }
      }
    } else {
      workload_array = c.workload_array;
      for_each_sched([&](Schedule& sched) {
        // replace this ssmodel with the copy
        sched.swap_model(_ssModel.subModel());
        sched.set_model(&_ssModel);
      });
    }

    unused_nodes = c.unused_nodes;
    unused_links = c.unused_links;
  }


  bool inputCreep(dfg::InputPort* input, ssivport* cand, sslink* skip) {
    int currentBit = 0;
    int currentLink = 0;

    if (currentLink >= cand->out_links().size())
      return false;

    // The number of lanes needed to map
    int lanes = input->vectorLanes();
    // The bitwidth of each lane
    int bitwidth = input->bitwidth();

    // Get the current link
    sslink* link = cand->out_links()[currentLink];

    // First Case when link is the skip link
    if (link == skip && cand->vp_stated()) {
      return false;
    }

    if (cand->vp_stated()) {
      if (link->bitwidth() < 8) {
        return false;
      }
      currentLink++;
      currentBit = 0;
      if (currentLink >= cand->out_links().size()) {
        return false;
      }
      link = cand->out_links()[currentLink];
    }

    // Creep through all the lanes
    for (int currentLane = 0; currentLane < lanes; currentLane++) {
      while (link->bitwidth() < currentBit + bitwidth) {
        currentLink++;
        currentBit = 0;
        if (currentLink >= cand->out_links().size()) {
          return false;
        }
        link = cand->out_links()[currentLink];
      }

      if (link == skip) {
        return false;
      }
      currentBit += bitwidth;
    }
    // Now we are at the end and it can map
    return true;
  }


  bool outputCreep(dfg::OutputPort* output, ssovport* cand, sslink* skip) {
    bool stated = output->penetrated_state >= 0;
    int currentBit = 0;
    int currentLink = 0;

    if (currentLink >= cand->in_links().size())
      return false;

    // Get the current link
    sslink* link = cand->in_links()[currentLink];
    // The number of lanes needed to map
    int lanes = output->vectorLanes();
    // The bitwidth of each lane
    int bitwidth = output->bitwidth();

    // First Case when link is the skip link
    if (link == skip && cand->vp_stated()) {
      return false;
    }

    // First add the stated edges if needed
    if (cand->vp_stated()) {
      if (link->bitwidth() < 8) {
        return false;
      }
      currentLink++;
      if (currentLink >= cand->in_links().size()) {
        return false;
      }
      link = cand->in_links()[currentLink];
    }


    // Creep through all the lanes
    for (int currentLane = 0; currentLane < lanes; currentLane++) {
      while (link->bitwidth() < currentBit + bitwidth) {
        currentLink++;
        currentBit = 0;
        if (currentLink >= cand->in_links().size()) {
          return false;
        }
        link = cand->in_links()[currentLink];
      }
      if (link == skip) {
        return false;
      }
      currentBit += bitwidth;
    }
    // Now we are at the end and it can map
    return true;
  }

  void check_vport_link_unassign(ssivport* ivport, sslink* link) {
    for_each_sched([&](Schedule& sched) {
      auto dfgnodes = sched.dfg_nodes_of(0, ivport);
      if (dfgnodes.size() != 1)
        return;

      auto input = dynamic_cast<dfg::InputPort*>(dfgnodes[0].first);

      if (!inputCreep(input, ivport, link)) {
        sched.unassign_dfgnode(dfgnodes[0].first); 
      }
    });
  }

  void check_vport_link_unassign(ssovport* ovport, sslink* link) {
    for_each_sched([&](Schedule& sched) {
      auto dfgnodes = sched.dfg_nodes_of(0, ovport);
      if (dfgnodes.size() != 1)
        return;
      auto output = dynamic_cast<dfg::OutputPort*>(dfgnodes[0].first);

      if (!outputCreep(output, ovport, link)) {
        sched.unassign_dfgnode(dfgnodes[0].first); 
      }
    });
  }

  /**
   * @brief Unassigns a link from all schedules
   * 
   * @param link the link to unassign
   */
  void unassign_link(sslink* link) {
    for_each_sched([&](Schedule& sched) {
      for (int slot = 0; slot < sched.num_slots(link); ++slot) {
        auto& edges = sched.dfg_edges_of(slot, link);
        for (auto it = edges.begin(); it != edges.end();) {
          
          auto& edge = sched.ssdfg()->edges[it->eid];

          sched.unassign_dfgnode(edge.def());
          sched.unassign_dfgnode(edge.use());
          //sched.unassign_edge(&edge);
        }
        DSA_CHECK(sched.link_prop()[link->id()].slots[slot].edges.empty());
      }
    });
  }

  bool check_link_used(sslink* link) {
    bool used = false;
    for_each_sched([&](Schedule& sched) {
      auto edges = sched.edge_prop();
      for (auto edge : edges) {
        for (auto edge_links : edge.links) {
          if (edge_links.second == link) {
            used = true;
            break;
          }
        }
      }
    });
    return used;
  }

  /**
   * @brief Unassigns a node from all schedules
   * 
   * @param node the node to unassign 
   */
  void unassign_node(ssnode* node) {
    for_each_sched([&](Schedule& sched) {
      for (int slot = 0; slot < sched.num_slots(node); ++slot) {
        auto& vertices = sched.dfg_nodes_of(slot, node);
        for (auto i = vertices.begin(); i != vertices.end();) {
          sched.unassign_dfgnode(i->first);
        }
        DSA_CHECK(sched.dfg_nodes_of(slot, node).empty());

        auto& passthroughs = sched.dfg_passthroughs_of(slot, node);
        for (auto i = passthroughs.begin(); i != passthroughs.end();) {
          sched.remove_passthrough_from_edge(i->first, {i->second, node});
        }
        DSA_CHECK(sched.dfg_passthroughs_of(slot, node).empty());
      }
    });
  }

  /**
   * @brief Checks whether vector port needs to change its stated value
   * 
   * @param node the node to check
   * @param deleting whether a link will be deleted from the schedule
   */
  void check_stated(ssnode* node) {
    if (auto vport = dynamic_cast<SyncNode*>(node)) {
      bool prev_stated = vport->vp_stated();
      if (vport->isInputPort() && vport->out_links().size() < 2 && !must_stay_stated(vport)) {
        vport->vp_stated(false);
      }
      if (vport->isOutputPort() && vport->in_links().size() < 2) {
        vport->vp_stated(false);
      }
      if (prev_stated != vport->vp_stated())
        unassign_node(vport);
    }
  }

  /**
   * @brief Adds a link to the ADG
   * 
   */
  sslink* add_link(ssnode* source, ssnode* sink, int souceSlot=-1, int sinkSlot=-1) {
    auto* sub = _ssModel.subModel();
    // The register engine never feeds input ports (see memory_feeds_inputs);
    // random link mutations must not create such a link either.
    if (dynamic_cast<ssregister*>(source) && dynamic_cast<ssivport*>(sink)) return nullptr;
    auto* link = sub->add_link(source, sink, souceSlot, sinkSlot);
    return link;
  }

  /**
   * @brief Delete a link on every schedule
   * 
   * @param link link to delete
   */
  void delete_link(sslink* link, bool unassign_vport=true) {
    //bool used = check_link_used(link);
    auto* sub = _ssModel.subModel();

    if (auto ivport = dynamic_cast<ssivport*>(link->source())) {
      if (unassign_vport) {
        check_vport_link_unassign(ivport, link);
        check_stated(link->source());
      }
    }

    if (auto ovport = dynamic_cast<ssovport*>(link->sink())) {
      if (unassign_vport) {
        check_vport_link_unassign(ovport, link);
        check_stated(link->sink());
      }
    }

    // remove it from every schedule, even if unassign=true
    unassign_link(link);

    sub->delete_link(link->id());

    for_each_sched([&](Schedule& sched) {
      sched.remove_link(link->id());
    });
    
    delete link;

    verify();
  }

  int check_overprovisioning() {
    int overprovisioned = 0;
    for (int i = 0; i < workload_array.size(); ++i) {
      auto& ws = workload_array[i];
      bool non_overprovedSchedule = false;
      for (Schedule& sched : ws.sched_array) {
        SchedStats s;
        sched.get_overprov(s.ovr, s.agg_ovr, s.max_util);
        if (s.ovr == 0) {
          non_overprovedSchedule = true;
          break;
        }
      }
      if (!non_overprovedSchedule) {
        overprovisioned++;
      }
    }
    return overprovisioned;
  }

  std::pair<int, std::vector<std::string>> check_schedules() {
    int num_not_working = 0;
    std::vector<std::string> not_working_schedules;
    for (int i = 0; i < workload_array.size(); ++i) {
      bool working_schedule = false;
      auto& ws = workload_array[i];
      for (Schedule& sched : ws.sched_array) {
        if (sched.is_complete<dfg::Node*>()) {
          working_schedule = true;
          break;
        }
      }
      if (!working_schedule) {
        num_not_working++;
        not_working_schedules.push_back(ws.sched_array[0].ssdfg()->filename);
      }
    }
    return std::make_pair(num_not_working, not_working_schedules);
  }

  double configuration_performance(int num_cores_local, int num_banks_local, int system_bus_width_local) {
    auto dma = _ssModel.subModel()->dma_list();
    int old_system_bus_size = 0;
    if (dma.size() > 0) {
      old_system_bus_size = dma[0]->readWidth();
      dma[0]->readWidth(system_bus_width_local);
      dma[0]->writeWidth(system_bus_width_local);
    }


    double total = 1.0;
    for (int i = 0; i < workload_array.size(); ++i) {
      auto& ws = workload_array[i];
      double score = (double)1e-3;     
      for (Schedule& sched : ws.sched_array) {
        std::string spm_performance = "";
        std::string l2_performance = "";
        std::string dram_read_performance = "";
        score = std::max(score, dse_sched_obj(&sched, num_cores_local, num_banks_local, spm_performance, l2_performance, dram_read_performance));
      }
      total *= score * weight[i];
    }

    // Restore old system bus size
    if (dma.size() > 0) {
      dma[0]->readWidth(old_system_bus_size);
      dma[0]->writeWidth(old_system_bus_size);
    }
    
    // Return harmonic mean of all schedules performances
    return pow(total, (1.0 / workload_array.size()));
  }

  double dse_sched_obj(Schedule* sched, int cores, int banks, std::string& spm_performance, std::string& l2_performance, std::string& dram_performance) {
    if (!sched) return 0.1;

    // YES, I KNOW THIS IS A COPY OF SCHED< JUST A TEST FOR NOW
    SchedStats s;
    int num_left = sched->num_left();
    bool succeed_sched = (num_left == 0);
    auto &delay_violation = sched->violation_penalty;
    delay_violation = std::make_pair(0, 0);

    sched->get_overprov(s.ovr, s.agg_ovr, s.max_util);
    sched->fixLatency(s.lat, s.latmis, delay_violation);

    int violation = sched->violation();
    double threads = cores;
    double performance = sched->estimated_performance(spm_performance, l2_performance, dram_performance, threads, banks);
    
    // Check to see that the performance actually exists
    if (performance > 1) {
      // Give the possibility to run at lower number of threads
      while (threads > 0) {
        threads -= 1;
        double thread_performance = sched->estimated_performance(spm_performance, l2_performance, dram_performance, threads, banks);
        if (thread_performance > performance) {
          performance = thread_performance;
        } else {
          break;
        }
      }
    }
    
    if (succeed_sched) {
      double eval = performance;
      if (delay_violation.second + delay_violation.first != 0)
        eval *= ((double)delay_violation.second  / (delay_violation.second + delay_violation.first));

      eval /= std::max(1.0 + s.ovr, sqrt(s.agg_ovr));

      return eval;
    }

    return -num_left;
  }

  double constrained_system_explore(dsa::adg::estimation::Result prev_estimation, int num_cores, int num_banks, int system_bus_width) {
    auto dma = _ssModel.subModel()->dma_list();
    int links = 0;
    if (dma.size() > 0) {
      links = dma[0]->in_links().size() + dma[0]->out_links().size();
    }

    auto resource_estimation = prev_estimation;
    resource_estimation.scale_cores(num_cores);
    resource_estimation.add_system_bus_overhead(num_cores, num_banks, system_bus_width);
    auto scaled = resource_estimation.sum();
    scaled->normalize();
    double constrained_resource = scaled->constrained_resource(0);

    delete scaled;
    return constrained_resource;
  }

  void system_bus_explore(int& num_cores, int& num_banks, int& system_bus_width) {
    double performance = -1;

    // Now we should calculate system-level parameters
    // The system bus width is not explored: configuration_performance() used to
    // leave the DMA at the last trial width (64 B), and any width other than the
    // seed's 32 B corrupts stream data on the generated RTL (8 B scratchpad bus,
    // 64 B DMA bus both observed). Keep whatever the seed specifies.
    std::vector<int> possible_system_bus = {32};
    {
      auto dma = _ssModel.subModel()->dma_list();
      if (!dma.empty()) possible_system_bus = {dma[0]->readWidth()};
    }

    for (int system_bus_width_local : possible_system_bus) {
      // Setup resource estimation
      auto resource_estimation = dsa::adg::estimation::EstimatePowerAera(&_ssModel);
      resource_estimation.add_core_overhead();

      // Search for best number of cores and banks together
      int num_cores_local = this->num_cores;
      int num_banks_local = next_power_of_two(num_cores_local);
      double constrained_resource_local = constrained_system_explore(resource_estimation, num_cores_local, num_banks_local, system_bus_width_local);
      // We need to iterate up the number of cores
      if (constrained_resource_local < 1.0) {
        double local_performance = configuration_performance(num_cores_local, num_banks_local, system_bus_width_local);

        if (local_performance > performance) {
          performance = local_performance;
          num_cores = num_cores_local;
          num_banks = num_banks_local;
          system_bus_width = system_bus_width_local;
        } else if (local_performance == performance) {
          if (num_cores_local > num_cores) {
            performance = local_performance;
            num_cores = num_cores_local;
            num_banks = num_banks_local;
            system_bus_width = system_bus_width_local;
          }
        }

        while (constrained_resource_local < 1.0) {
          num_cores_local += 1;
          num_banks_local = next_power_of_two(num_cores_local);
          constrained_resource_local = constrained_system_explore(resource_estimation, num_cores_local, num_banks_local, system_bus_width_local);
          if (constrained_resource_local < 1.0) {
            local_performance = configuration_performance(num_cores_local, num_banks_local, system_bus_width_local);
            if (local_performance > performance) {
              performance = local_performance;
              num_cores = num_cores_local;
              num_banks = num_banks_local;
              system_bus_width = system_bus_width_local;
            } else if (local_performance == performance) {
              if (num_cores_local > num_cores) {
                performance = local_performance;
                num_cores = num_cores_local;
                num_banks = num_banks_local;
                system_bus_width = system_bus_width_local;
              }
            }
          }
        }
      } else {
        // We need to go down with number of cores;
        while (constrained_resource_local >= 1.0 && num_cores_local > 0) {
          num_cores_local -= 1;
          num_banks_local = next_power_of_two(num_cores_local);
          constrained_resource_local = constrained_system_explore(resource_estimation, num_cores_local, num_banks_local, system_bus_width_local);
          if (constrained_resource_local < 1.0) {
            double local_performance = configuration_performance(num_cores_local, num_banks_local, system_bus_width_local);
            if (local_performance > performance) {
              performance = local_performance;
              num_cores = num_cores_local;
              num_banks = num_banks_local;
              system_bus_width = system_bus_width_local;
            } else if (local_performance == performance) {
              if (num_cores_local > num_cores) {
                performance = local_performance;
                num_cores = num_cores_local;
                num_banks = num_banks_local;
                system_bus_width = system_bus_width_local;
              }
            }
          }
        }
      }
    }
  }

  std::pair<double, double> dse_obj(bool cache_obj=true) {
    if (objective.first > 0) {
      return objective;
    }
    double total_score = 1.0;
    res.resize(workload_array.size());

    int best_system_bus_size = -1;
    int best_num_cores = -1;
    int best_num_banks = -1;

    system_bus_explore(best_num_cores, best_num_banks, best_system_bus_size);

    if (num_cores == -1) {
      dse_fail_reason = "budget";
      return std::make_pair<double, double>(1.5, 0.0); 
    }

    num_cores = best_num_cores;
    num_banks = best_num_banks;
    system_bus = best_system_bus_size;
    for (auto dma : ss_model()->subModel()->dma_list()) {
      dma->writeWidth(system_bus);
      dma->readWidth(system_bus);
    }


    std::vector<std::vector<std::pair<std::string, double>>> _dfg_performances;
    std::vector<std::vector<std::pair<std::string, std::string>>> _dfg_spm_performances;
    std::vector<std::vector<std::pair<std::string, std::string>>> _dfg_l2_performances;
    std::vector<std::vector<std::pair<std::string, std::string>>> _dfg_dram_performances;
    std::vector<double> _workload_performances;
    std::vector<double> _workload_weights;

    for (int i = 0; i < workload_array.size(); ++i) {
      auto& ws = workload_array[i];
      double score = (double)1e-3;
      res[i] = nullptr;
      std::vector<std::pair<std::string, double>> performances;
      std::vector<std::pair<std::string, std::string>> spm_performance;
      std::vector<std::pair<std::string, std::string>> dram_performance;
      std::vector<std::pair<std::string, std::string>> l2_performance;        
      for (Schedule& sched : ws.sched_array) {
        
        std::string spm_performance_local = "";
        std::string l2_performance_local = "";
        std::string dram_performance_local = "";
        
        double new_score = dse_sched_obj(&sched, num_cores, num_banks, spm_performance_local, l2_performance_local, dram_performance_local);

        if (new_score >= score) {
          score = new_score;
          res[i] = &sched;
        }
        performances.push_back(std::make_pair(sched.ssdfg()->filename, new_score));

        spm_performance.push_back(std::make_pair(sched.ssdfg()->filename, spm_performance_local));
        l2_performance.push_back(std::make_pair(sched.ssdfg()->filename, l2_performance_local));
        dram_performance.push_back(std::make_pair(sched.ssdfg()->filename, dram_performance_local));
      }
      _dfg_performances.push_back(performances);
      _dfg_spm_performances.push_back(spm_performance);
      _dfg_l2_performances.push_back(l2_performance);
      _dfg_dram_performances.push_back(dram_performance);
      _workload_performances.push_back(score);
      _workload_weights.push_back(weight[i]);
      total_score *= std::pow(score, weight[i]);
    }

    dfg_performances = _dfg_performances;
    dfg_spm_performances = _dfg_spm_performances;
    dfg_l2_performances = _dfg_l2_performances;
    dfg_dram_performances = _dfg_dram_performances;
    workload_weights = _workload_weights;
    workload_performances = _workload_performances;

    // First Check to make sure that everything schedules
    std::pair<int, std::vector<std::string>> num_not_scheduled = check_schedules();
    if (num_not_scheduled.first != 0) {
      std::ostringstream s;
      s << std::to_string(num_not_scheduled.first) << " schedules not working (";
      for (auto& dfg : num_not_scheduled.second) {
        s << dfg << " ";
      }
      s << ")";


      dse_fail_reason = s.str();

      return std::make_pair<double, double>(1.0, 0.0);
    }

    // Now Check to make sure that everything is not overprovisioned
    int num_overproved = check_overprovisioning();
    if  (num_overproved != 0) {
      dse_fail_reason = std::to_string(num_overproved) + " schedules overprovisioned";
      // The objective should be slightly bigger than failing schedules
      return std::make_pair<double, double>(1.25, 0.0);
    }

    double total_weights = std::accumulate(workload_weights.begin(), workload_weights.end(), 0.0);
    total_score = pow(total_score, (1.0 / total_weights));
    performance = total_score;


    auto resource_estimation = dsa::adg::estimation::EstimatePowerAera(&_ssModel);
    resource_estimation.add_core_overhead();
    //resource_estimation.scale_cores(num_cores);
    //resource_estimation.add_system_bus_overhead(num_cores, num_banks, system_bus);
    auto scaled = resource_estimation.sum();
    scaled->normalize();

    normalized_resources = scaled->constrained_resource(0);

    // Set final score
    std::pair<double, double> final_score = std::make_pair(performance, normalized_resources);

    dse_fail_reason = "succeed";
    delete scaled;

    if (cache_obj)
      objective = final_score;

    return final_score;
  }

  std::pair<double, double> weight_obj() {
    return dse_obj();
  }


  /**
   * @brief Gets the utilization ratio of the current codesign instance
   * 
   * @return std::tuple<float, float, float>, the overall, node, and
   * link utilization ratio, respectively of current codesign instance
   */
  std::tuple<float, float, float> utilization() {
    // If DSE_OBJ is horrible return nothing
    if (abs(dse_obj(false).first) < 2) {
      return {0, 0, 0};
    }
    
    // Check to see if there is a working schedule
    bool meaningful = false;
    for (auto sched : res) {
      if (sched != nullptr) {
        meaningful = true;
        break;
      }
    }

    // If there is no schedules, return nothing
    if (!meaningful) {
      return {0, 0, 0};
    }

    auto sub = _ssModel.subModel();
    

    unused_nodes = std::vector<bool>(sub->node_list().size(), true);
    unused_links = std::vector<bool>(sub->link_list().size(), true);

    for (size_t i = 0; i < res.size(); ++i) {
      auto sched = res[i];
      if (sched != nullptr) {
        for (auto& ep : sched->edge_prop()) {
          for(auto link_iter = ep.links.begin(); link_iter < ep.links.end(); link_iter++) {
            auto link = *link_iter;
            unused_links[link.second->id()] = false;
            unused_nodes[link.second->sink()->id()] = false;
            unused_nodes[link.second->source()->id()] = false;
          }
        }
      }
    }

    float node_count = unused_nodes.size();
    float link_count = unused_links.size();
    float unused_node_count = 0;
    float unused_link_count = 0;

    // Go through and count how many are true
    for (int i = 0; i < unused_nodes.size(); i++) {
      if (!unused_nodes[i]) unused_node_count++;
    }

    for (int i = 0; i < unused_links.size(); i++) {
      if (!unused_links[i]) unused_link_count++;
    }

    // do calculations to determine unitilization
    float overall_usage = (unused_node_count + unused_link_count) / (node_count + link_count);
    float nodes_usage = unused_node_count / node_count;
    float links_usage = unused_link_count / link_count;
    return {overall_usage, nodes_usage, links_usage};
  }

  /**
   * @brief Checks if a stated Edge is mapped to a vector port
   * 
   * @param sync the vector port to check
   * @return true if a stated edge is mapped to this vector port
   * @return false if a stated edge is not mapped to this vector port
   */
  bool stated_mapped(SyncNode* sync) {
    for (size_t i = 0; i < res.size(); ++i) {
      auto sched = res[i];
      if (sched != nullptr) {
        // Loop through all the edges
        std::vector<dsa::dfg::Edge> edges = sched->ssdfg()->edges;
        for (int j = 0; j < sched->edge_prop().size(); j++) {
          auto& ep = sched->edge_prop()[j];
          if (ep.links.size() == 0)
              continue;
          auto& edge = edges[j];
          if (sync->isInputPort() && edge.sourceStated()) {
            // Only need to check first link
            ssnode* edge_source = ep.links[0].second->source();
            if (edge_source->id() == sync->id())
              return true;
            
          } else if (sync->isOutputPort() && edge.sinkStated()) {
            // Only need to check last link as they are ordered
            ssnode* edge_sink = ep.links[ep.links.size() - 1].second->sink();
            if (edge_sink->id() == sync->id())
              return true;
          }
        }
      }
    }
    return false;
  }

  int countSchedules() {
    int count = 0;
    for (auto workload : workload_array) {
      for (auto sched : workload.sched_array) {
        count++;
      }
    }
    return count;
  }

  void dump_breakdown(bool verbose) {
    if (verbose) {
      auto estimated = dsa::adg::estimation::EstimatePowerAera(ss_model());
      estimated.Dump(std::cout);
      for (int i = 0, n = res.size(); i < n; ++i) {
        if (!res[i]) {
          std::cout << "[skip]" << std::endl;
          continue;
        }
        std::string spm_performance = "";
        std::string dram_performance = "";
        std::string l2_performance = "";
        auto performance = dse_sched_obj(res[i], num_cores, num_banks, spm_performance, l2_performance, dram_performance);
        std::cout << res[i]->ssdfg()->filename << ": " << performance << std::endl;
      }
    }
  }

  /**
   * @brief Checks to see if a flip-flop can be removed without causing
   * a cycle in the hardware graph
   * 
   * @param n the node to check 
   * @return true if there will be no cycle created by removing the node
   * @return false if there will be a cycle created by removing the node
   */
  bool check_cycle(ssnode* n) {
    std::vector<bool> visited(_ssModel.subModel()->node_list().size(), false);

    // Check upstream for cycles
    if (!check_cycle_helper(n, visited, true))
      return false;
    
    // Check downstream for cycles
    if (!check_cycle_helper(n, visited, false))
      return false;
    
    // No cycles found, so removing flipflow is fine
    return true;
  }

 private:
  /**
  * @brief Helper for check_cycle. Will recursively check all nodes
  * to see if there is a cycle
  * 
  * @param n the node to check
  * @param visited the set of nodes that have been visited
  * @param down whether to check the upstream or downstream
  * @return true if there is no cycle from removing this node
  * @return false if there is a cycle from removing this node
  */
  bool check_cycle_helper(ssnode* n, std::vector<bool>& visited, bool down) {
    auto links = n->in_links();
    if (down) {
      links = n->out_links();
    }

    for (auto link : links) {
      auto node = link->source();
      if (down) {
        node = link->sink();
      }
      
      // If there is a delay then we can just continue
      if (node->max_delay() > 0) 
        continue;

      // If we have already visited this node then we have a cycle
      if (visited[node->id()])
        return false;
      visited[node->id()] = true;

      // recursively check all the nodes connected to this node
      if (!check_cycle_helper(node, visited, down))
        return false;
    }
    // No cycle found
    return true;
  }
  

  
  /**
   * @brief Collapses a vport
   * TODO Fix this
   * @param vport vport to collapse
   */
  void collapse_vport(SyncNode* vport) {
    // Get Links used in schedules
    std::unordered_set<sslink*> links = useful_vport_links(vport);
    // If no links are used in any schedule, we can just remove vport
    if (links.size() == 0) return;

    auto* sub = _ssModel.subModel();
    SyncNode* other_vport;
    
    if (vport->out_links().size() == 0) {
      // Find another random VPort to collapse this one into
      while (other_vport == nullptr || other_vport->id() == vport->id()) {
        int index = non_uniform_random(sub->output_list(), unused_nodes);
        other_vport = sub->output_list()[index];
      }
      // Check if width will go up by more than power of two
      if (log2(other_vport->in_links().size()) == log2(links.size() + other_vport->in_links().size())) {
        // Simply add links to this vport
        for (auto& link : links) {
          add_link(link->source(), other_vport);
        }
      } else {
        // Add Switches so VPort retains current power of two width
        if (other_vport->in_links().size() > links.size()) {
          for (auto& link : other_vport->in_links()) {
            if (links.size() > 0) {
              auto link_to_add = links.begin();
              ssswitch* sw = add_switch_like_fabric();
              sub->add_link(sw, other_vport);
              sub->add_link(link->source(), sw);
              sub->add_link((* link_to_add)->source(), sw);
              delete_link(link);
              links.erase(link_to_add);
            }
          }
        } else {
          // First add
          for (auto& link : other_vport->in_links()) {
            ssswitch* sw = add_switch_like_fabric();
            sub->add_link(sw, other_vport);
            sub->add_link(link->source(), sw);
            delete_link(link);
          }
          // Now add to the switches
          int link_index = 0;
          for (auto& link : links) {
            sub->add_link(link->source(), other_vport->in_links()[link_index]->source());
            link_index++;
            if (link_index == other_vport->in_links().size()) {
              link_index = 0;
            }
          }
        }
      }
    } else {
      // Find another random VPort to collapse this one into
      while (other_vport == nullptr || other_vport->id() == vport->id()) {
        int index = non_uniform_random(sub->input_list(), unused_nodes);
        other_vport = sub->input_list()[index];
      }
      // Check if width will go up by more than power of two
      if (log2(other_vport->out_links().size()) == log2(links.size() + other_vport->out_links().size())) {
        // Simply add links to this vport
        for (auto& link : links) {
          sub->add_link(other_vport, link->sink());
        }
      } else {
        // Add Switches so VPort retains current power of two width
        if (other_vport->out_links().size() > links.size()) {
          for (auto& link : other_vport->out_links()) {
            if (links.size() > 0) {
              auto link_to_add = links.begin();
              ssswitch* sw = add_switch_like_fabric();
              sub->add_link(other_vport, sw);
              sub->add_link(sw, link->sink());
              sub->add_link(sw, (* link_to_add)->sink());
              delete_link(link);
              links.erase(link_to_add);
            }
          }
        } else {
          // First add
          for (auto& link : other_vport->out_links()) {
            ssswitch* sw = add_switch_like_fabric();
            sub->add_link(sw, other_vport);
            sub->add_link(sw, link->sink());
            delete_link(link);
          }
          // Now add to the switches
          int link_index = 0;
          for (auto& link : links) {
            sub->add_link(other_vport->out_links()[link_index]->sink(), link->sink());
            link_index++;
            if (link_index == other_vport->out_links().size()) {
              link_index = 0;
            }
          }
        }
      }
    }
    std::ostringstream s;
    s << dse_changes_log.back() << " collapsed vport into " << other_vport->name();
    dse_changes_log.back() = s.str();
  }

  std::unordered_set<sslink*> useful_vport_links(SyncNode* n) {
    std::unordered_set<sslink*> links;
    for_each_sched([&](Schedule& sched) {
      auto link_prop = sched.link_prop(); 
      //Output Vector Port
      if (n->out_links().size() == 0) {
        for (auto& link : n->in_links()) {
          auto link_slice = link_prop[link->id()];
          if (link_slice.slots.size() > 0) {
            links.insert(link);
          }
        }
      } else {
        for (auto& link : n->out_links()) {
          auto link_slice = link_prop[link->id()];
          if (link_slice.slots.size() > 0) {
            links.insert(link);
          }
        }
      }
    });
    return links;
  }

  /**
   * @brief Collapses the links through a node, of a given edge
   * 
   * @param n node to collapse
   * @param edge edge to collapse from
   * @param sched currently used schedule
   */
  void collapse_edge_links(ssnode* n, Schedule::EdgeProp edgeProp, Schedule& sched, std::vector<sslink*>& collapsed_links) {
    auto* sub = _ssModel.subModel();
    auto& links = edgeProp.links;
    // loop through links
    for (int i = 0; i < links.size(); i++) {
      sslink* link  = links[i].second;
      //Check if link is the node to collapse
      if (link->sink()->id() == n->id() && i + 1 < links.size()) {
        auto src = link->source();
        sslink* next_link  = links[i + 1].second;
        auto dst = next_link->sink();
        // Append the bypass link: inserting it at the positions of the two
        // links it replaces gives it two different "creation times", which
        // breaks the per-node link order dsagen2 derives port indices from.
        sslink* collapsed_link = add_link(src, dst);

        // Repair Schedule
        
        if (collapsed_link != nullptr)
          collapsed_links.push_back(collapsed_link);
        return;
      }
    }
  }

  void collapse_and_repair(ssnode* n, std::vector<sslink*> &collapsed_links) {
    for_each_sched([&](Schedule& sched) {
      bool not_collapsed = false;
      auto& edge_prop = sched.edge_prop();
      for (int i = 0; i < edge_prop.size(); i++) {
        auto& edge = edge_prop[i];
        for (int j = 0; j < edge.links.size(); j++) {
          auto& link = edge.links[j];
          if (link.second->sink()->id() != n->id()) {
            continue;
          }
          if (j + 1 == edge.links.size()) {
            not_collapsed = true;
            continue;
          }
          DSA_LOG(COLLAPSE) << "Collapsing node " << n->name() << " with link " << link.second->name() << " and link " << edge.links[j + 1].second->name() << " on edge " << i;
          
          ssnode* src = link.second->source();
          auto next_link  = edge.links[j + 1];
          ssnode* dst = next_link.second->sink();
          // See collapse_edge_links: the bypass link is appended, not
          // inserted at the replaced links' positions.
          sslink* collapsed_link = add_link(src, dst);
          if (collapsed_link == nullptr)
            continue;

          // Allocate space for link
          sched.allocate_space();
          
          /*
          // Now repair schedule

          // First Remove old links from schedule
          sched.remove_link_from_edge(&sched.ssdfg()->edges[i], link);
          sched.remove_link_from_edge(&sched.ssdfg()->edges[i], next_link);

          // Remove passthrough if functional unit
          for (auto& passthrough : edge.passthroughs) {
            if (passthrough.second->id() == n->id()) {
              sched.remove_passthrough_from_edge(&sched.ssdfg()->edges[i], passthrough);
            }
          }

          // Now add new link to schedule
          sched.assign_edgelink(&sched.ssdfg()->edges[i], src_slot, collapsed_link, j);
          collapsed_links.push_back(collapsed_link);
          */
        }
      }
      return;
    });
  }


  /**
   * @brief Schedule-preserving Transformation for Stated Vector Ports
   * As a stated link must be used as the first link, if we are removing
   * the stated property then we must also remove the first link to preserve
   * the schedule.
   * 
   * @param n the vector port to collapse the stated property
   * @param preserve_schedule whether to preserve the schedulability
   * This property is important for prunning which requires schedulability
   * remains
   */
  void stated_collapse(ssnode* n) {
    auto* sub = _ssModel.subModel();
    if (auto sync = dynamic_cast<SyncNode*>(n)) {
      if (sync->vp_stated() && !must_stay_stated(sync)) {
        // Change VP_Stated to false
        sync->vp_stated(false);
        
        // If it doesn't use a crossbar then the first link must be deleted
        // Otherwise we should delete a random link
        if (sync->isInputPort()) {
          delete_link(sync->out_links()[0], false);
        } else {
          delete_link(sync->in_links()[0], false);
        }

        for_each_sched([&](Schedule& sched) {
          auto dfgnodes = sched.dfg_nodes_of(0, n);

          if (dfgnodes.size() != 1)
            return;

          if (auto input = dynamic_cast<dfg::InputPort*>(dfgnodes[0].first)) {
            if (input->stated) {
              sched.unassign_dfgnode(dfgnodes[0].first); 
            }
          } else if (auto output = dynamic_cast<dfg::OutputPort*>(dfgnodes[0].first)) {
            bool stated = output->penetrated_state >= 0;
            if (stated) {
              sched.unassign_dfgnode(dfgnodes[0].first); 
            }
          } else {
            DSA_CHECK(false) << "Unknown node type";
          }
        });
      }
    }
  }

  /**
   * @brief Adds a fifo depth to all nodes after a given node in the given edge
   * 
   * TODO: Only Check the last node in an endge
   * 
   * @param fifos set of fu's that have already have their fifo depth added
   * @param n node to collapse
   * @param edge edge to check for functional units
   * @param sched current schedule to use
   */
  void add_fifo_edge(std::unordered_set<ssfu*>& fifos, ssnode* n, Schedule::EdgeProp edge, Schedule& sched) {

    bool found_node = false;
    for(auto link : edge.links) {
      if (found_node) {
        if (auto fu = dynamic_cast<ssfu*>(link.second->sink())) {
          // Check if node is already in the set or if it has too large a delay fifo
          if (fu->delay_fifo_depth() < 32 && fifos.find(fu) == fifos.end()) {
            fu->max_delay(fu->delay_fifo_depth() + 1);
            fifos.insert(fu);
            // We don't need to reassign these to the schedule

            return;
          }
        }
      } else if (link.second->sink()->id() == n->id()) {
        found_node = true;
      }
    }
    /*
    for (auto link : sched.links_of(&edge)) {
      if (found_node) {
        if (auto fu = dynamic_cast<ssfu*>(link.second->sink())) {
          // Check if node is already in the set or if it has too large a delay fifo
          if (fu->delay_fifo_depth() < 32 && fifos.find(fu) == fifos.end()) {
            fu->max_delay(fu->delay_fifo_depth() + 1);
            fifos.insert(fu);
            // We don't need to reassign these to the schedule
            return;
          }
        }
      } else if (link.second->sink()->id() == n->id()) {
        found_node = true;
      }
    }
    */
  }
  
  /**
   * @brief Deletes a node
   * 
   * When we delete a hardware element, we need to:
   *  1. deschedule anything that was assigned to that element
   *  2. remove the concept of that element from the schedule (consistency)
   *  3. remove the element from the hardware description
   * 
   * @param n node to delete
   * @param collapse_links whether to collapse the links of node
   * @param forward_vport whether to forward the links of vport
   * @param add_fifo whether to add fifo depths to all nodes after
   */
  void delete_node(ssnode* n, bool collapse_links=false, bool forward_vport=false, bool add_fifo=false) {
    auto* sub = _ssModel.subModel();
    
    std::unordered_set<ssfu*> fifos;
    std::vector<sslink*> collapsed_links;

    if (collapse_links) {
      collapse_and_repair(n, collapsed_links);
    }
    
    
    for_each_sched([&](Schedule& sched) {
      auto edges = sched.ssdfg()->edges;
      for (int i = 0; i < sched.edge_prop().size(); ++i) {
        auto edgeProp = sched.edge_prop()[i];
        if (add_fifo && sched.edge_delay(&edges[i]) < 1) {
          add_fifo_edge(fifos, n, edgeProp, sched);
        }
      }
    });

    for_each_sched([&](Schedule& sched) { sched.allocate_space(); });

    while (n->in_links().size() > 0) {
      delete_link(n->in_links()[0]);
    }

    while (n->out_links().size() > 0) {
      delete_link(n->out_links()[0]);
    }

    unassign_node(n);
    for_each_sched([&](Schedule& sched) { sched.ensure_node_delete(n->id()); });

    sub->delete_node(n->id());
    for_each_sched([&](Schedule& sched) { sched.remove_node(n->id()); });

    if (collapsed_links.size() > 0) {
      std::ostringstream s;
      s << dse_changes_log.back() << " collapsed " << "(" << collapsed_links.size() << ") ";
      
      for (int i = 0; i < collapsed_links.size(); ++i) {
        int inIndex = collapsed_links[i]->source()->link_index(collapsed_links[i], false);
        int inSize = collapsed_links[i]->source()->out_links().size();
        int outIndex = collapsed_links[i]->sink()->link_index(collapsed_links[i], true);
        int outSize = collapsed_links[i]->sink()->in_links().size();
        s << collapsed_links[i]->name() << " " << inIndex << "(" << inSize << ")" << "->" << outIndex << "(" << outSize << ") ";
        if (i < collapsed_links.size() - 1) {
          s << ", ";
        }
      }
      
      dse_changes_log.back() = s.str();
    }


    if (fifos.size() > 0) {
      std::ostringstream s;
      s << dse_changes_log.back() << " fifo depth added for ";
      for(auto it = fifos.begin(); it != fifos.end(); ++it) {
        s << "FU_" << (*it)->id() << " (" << (*it)->delay_fifo_depth() << ")";
        if (std::next(it) != fifos.end())
          s << ",";
      }
      dse_changes_log.back() = s.str();
    }

    delete n;
    verify();
  }
};

namespace dsa {
void DesignSpaceExploration(CodesignInstance* cur_ci);
} // namespace dsa
