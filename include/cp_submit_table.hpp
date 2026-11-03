#ifndef CP_SUBMIT_TABLE_HPP
#define CP_SUBMIT_TABLE_HPP

#include <map>
#include <mutex>

/* Submits still waiting for the pool's reply, by JSON-RPC request id, with the time each was
 * sent. An entry leaves on its own reply, on a failed send, or when the connection is
 * dropped; the pool reader drops the connection once the oldest entry has waited too long,
 * so the table only ever holds the submits of the last timeout window. Thread-safe. */
class CpSubmitTable {
public:
    void track(int id, double now)
    {
        std::lock_guard<std::mutex> lk(mx_);
        sent_[id] = now;
    }
    /* 1 when id was waiting (it no longer is), 0 for an unknown id. */
    int finish(int id)
    {
        std::lock_guard<std::mutex> lk(mx_);
        return sent_.erase(id) != 0;
    }
    int pending(int id) const
    {
        std::lock_guard<std::mutex> lk(mx_);
        return sent_.count(id) != 0;
    }
    /* Send time of the oldest waiting submit; 0 when none waits. */
    int oldest(double* sent_at) const
    {
        std::lock_guard<std::mutex> lk(mx_);
        if(sent_.empty()) return 0;
        double t = sent_.begin()->second;
        for(const auto& entry : sent_)
            if(entry.second < t) t = entry.second;
        *sent_at = t;
        return 1;
    }
    void clear()
    {
        std::lock_guard<std::mutex> lk(mx_);
        sent_.clear();
    }

private:
    mutable std::mutex mx_;
    std::map<int, double> sent_;
};

#endif
