#pragma once

#include <map>
#include <mutex>

/* Connection-local deadlines use monotonic seconds supplied by the caller.
 * Tests can exercise timeout boundaries without waiting for a real pool. */
class CpPoolSession {
public:
    enum class Timeout { None, Authorize, FirstJob, Submit };
    static constexpr double kAuthorizeTimeout = 30.0;
    static constexpr double kFirstJobTimeout = 30.0;
    static constexpr double kSubmitTimeout = 60.0;

    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        stage_ = Stage::Idle;
        auth_id_ = -1;
        first_job_ = false;
        submits_.clear();
    }

    void begin_authorize(int id, double now) {
        std::lock_guard<std::mutex> lock(mutex_);
        auth_id_ = id;
        stage_ = Stage::Authorize;
        deadline_ = now + kAuthorizeTimeout;
        first_job_ = false;
    }

    bool authorize_response(int id, bool accepted, double now) {
        std::lock_guard<std::mutex> lock(mutex_);
        if(stage_ != Stage::Authorize || id != auth_id_) return false;
        auth_id_ = -1;
        stage_ = accepted ? (first_job_ ? Stage::Ready : Stage::FirstJob) : Stage::Rejected;
        deadline_ = now + kFirstJobTimeout;
        return true;
    }

    void received_job() {
        std::lock_guard<std::mutex> lock(mutex_);
        first_job_ = true;
        if(stage_ == Stage::FirstJob) stage_ = Stage::Ready;
    }

    bool authorized() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return stage_ == Stage::FirstJob || stage_ == Stage::Ready;
    }

    bool begin_submit(int id, double now) {
        std::lock_guard<std::mutex> lock(mutex_);
        /* Bound memory if a pool reads shares but never acknowledges them. */
        return submits_.size() < 4096 && submits_.emplace(id, now).second;
    }

    bool finish_submit(int id) {
        std::lock_guard<std::mutex> lock(mutex_);
        return submits_.erase(id) != 0;
    }

    bool has_pending_submits() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return !submits_.empty();
    }

    Timeout expired(double now) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if(stage_ == Stage::Authorize && now >= deadline_) return Timeout::Authorize;
        if(stage_ == Stage::FirstJob && now >= deadline_) return Timeout::FirstJob;
        for(const auto& submit : submits_)
            if(now - submit.second >= kSubmitTimeout) return Timeout::Submit;
        return Timeout::None;
    }

private:
    enum class Stage { Idle, Authorize, FirstJob, Ready, Rejected };
    mutable std::mutex mutex_;
    Stage stage_ = Stage::Idle;
    int auth_id_ = -1;
    double deadline_ = 0.0;
    bool first_job_ = false;
    std::map<int, double> submits_;
};
