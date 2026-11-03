/* Before any include: a header that pulls in <cassert> first would keep assert off in Release. */
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "cp_cli.h"

#include <cassert>
#include <string>
#include <vector>

template<class F>
static bool rejects(F f)
{
    try {
        f();
    } catch(const std::invalid_argument&) {
        return true;
    }
    return false;
}

/* The message f throws, or "" if it does not throw. */
template<class F>
static std::string message(F f)
{
    try {
        f();
    } catch(const std::invalid_argument& error) {
        return error.what();
    }
    return "";
}

static const char* value_of(std::vector<std::string> args, int index, size_t capacity = 0,
                            int* next = nullptr)
{
    static std::vector<std::string> storage;
    static std::vector<char*> argv;
    storage = args;
    argv.clear();
    for(auto& arg : storage) argv.push_back(&arg[0]);
    const char* value = cp_cli_value((int)argv.size(), argv.data(), index, capacity);
    if(next) *next = index;
    return value;
}

int main()
{
    assert(cp_cli_option("--batch-size", "--batch-size"));
    assert(cp_cli_option("--batch-size=4", "--batch-size"));
    assert(!cp_cli_option("--batch-size-extra", "--batch-size"));
    assert(!cp_cli_option("--m", "--max-nonce") && cp_cli_option("--m=2", "--m"));
    assert(cp_cli_option_name("--batch-size=4") == "--batch-size");
    assert(cp_cli_option_name("--verify") == "--verify");

    int next = 0;
    assert(std::string(value_of({"miner", "--threads", "4"}, 1, 0, &next)) == "4" && next == 2);
    assert(std::string(value_of({"miner", "--threads=4"}, 1, 0, &next)) == "4" && next == 1);
    assert(std::string(value_of({"miner", "--mock-diff=-1"}, 1)) == "-1");
    assert(std::string(value_of({"miner", "--mock-diff", "-1"}, 1)) == "-1");
    assert(std::string(value_of({"miner", "--worker", "-rig01"}, 1)) == "-rig01");
    assert(message([] { value_of({"miner", "--pool"}, 1); }) == "requires a value");
    assert(rejects([] { value_of({"miner", "--pool", "--verify"}, 1); }));
    assert(rejects([] { value_of({"miner", "--profile-scan="}, 1); }));
    assert(message([] { value_of({"miner", "--worker", std::string(64, 'w')}, 1, 64); }) ==
           "value is longer than 63 characters");
    assert(std::string(value_of({"miner", "--worker", std::string(63, 'w')}, 1, 64)).size() == 63);

    {
        std::vector<std::string> storage = {"miner", "--worker", "rig02"};
        std::vector<char*> argv;
        for(auto& arg : storage) argv.push_back(&arg[0]);
        char worker[8] = "x";
        int i = 1;
        cp_cli_string((int)argv.size(), argv.data(), i, worker, sizeof(worker));
        assert(std::string(worker) == "rig02" && i == 2);
        storage[2] = "too-long-name";
        argv[2] = &storage[2][0];
        i = 1;
        assert(rejects([&] { cp_cli_string((int)argv.size(), argv.data(), i, worker, sizeof(worker)); }));
        assert(std::string(worker) == "rig02");
    }

    assert(cp_cli_int("0") == 0 && cp_cli_int("2147483647") == 2147483647);
    assert(cp_cli_int("3", 1, 3) == 3);
    for(const char* bad : {"", "-1", "+1", " 1", "1 ", "1junk", "abc", "2147483648", "99999999999"})
        assert(rejects([bad] { cp_cli_int(bad); }));
    assert(rejects([] { cp_cli_int("0", 1); }));
    assert(message([] { cp_cli_int("4", 1, 3); }) == "expected an integer from 1 to 3, got 4");
    assert(message([] { cp_cli_int("abc", 1); }) == "expected an integer of at least 1, got \"abc\"");

    assert(cp_cli_real("44", 1.0) == 44.0 && cp_cli_real("1e2", 1.0) == 100.0);
    assert(cp_cli_real("1.5", 1.0) == 1.5 && cp_cli_real("-0.5", -1.0) == -0.5);
    for(const char* bad : {"", "nan", "inf", "1e309", "0x10", "+1", "1.", ".5", "1e", "01", "1,2", " 1"})
        assert(rejects([bad] { cp_cli_real(bad, 0.0); }));
    assert(message([] { cp_cli_real("0.5", 1.0); }) ==
           "expected a decimal number of at least 1, got \"0.5\"");

    int m = 0, n = 0, mm = 0, mn = 0;
    assert(cp_cli_dimensions("4x8", &m, &n) == 2 && m == 4 && n == 8);
    assert(cp_cli_dimensions("4x8/64x64", &m, &n, &mm, &mn) == 4 && mm == 64 && mn == 64);
    for(const char* bad : {"4", "4x", "x8", "4x8x", "4x8/", "4x8/64", "4x8/64x", "4/8x8", "0x8", "4x8junk"})
        assert(rejects([bad, &m, &n, &mm, &mn] { cp_cli_dimensions(bad, &m, &n, &mm, &mn); }));
    assert(rejects([&m, &n] { cp_cli_dimensions("4x8/64x64", &m, &n); }));

    char host[16];
    assert(cp_cli_pool_uri("stratum+tcp://pool.example:3333", host, sizeof(host)) == 3333 &&
           std::string(host) == "pool.example");
    assert(cp_cli_pool_uri("stratum+tcp://pool.example:3333/", host, sizeof(host)) == 3333);
    assert(cp_cli_pool_uri("tcp://h:1/path", host, sizeof(host)) == 1 && std::string(host) == "h");
    for(const char* bad : {"pool.example:3333", "stratum+tcp://pool.example", "tcp://:3333",
                           "tcp://h:", "tcp://h:abc", "tcp://h:0", "tcp://h:65536", "tcp://h:99999",
                           "tcp://h:3333x", "tcp://h/x:3333", "tcp://1234567890123456:1"})
        assert(rejects([bad, &host] { cp_cli_pool_uri(bad, host, sizeof(host)); }));
    assert(cp_cli_pool_uri("tcp://123456789012345:1", host, sizeof(host)) == 1);
    assert(cp_cli_pool_uri("stratum+tcp://[::1]:3333", host, sizeof(host)) == 3333 &&
           std::string(host) == "::1");
    assert(cp_cli_pool_uri("tcp://[2001:db8::7]:1/", host, sizeof(host)) == 1 &&
           std::string(host) == "2001:db8::7");
    for(const char* bad : {"tcp://[::1]", "tcp://[::1]3333", "tcp://[]:1", "tcp://[::1:1",
                           "tcp://[::1]:", "tcp://[::1]:0"})
        assert(rejects([bad, &host] { cp_cli_pool_uri(bad, host, sizeof(host)); }));

    int devices[3], count = 0;
    cp_cli_devices("0,2", devices, count, 3);
    assert(count == 2 && devices[0] == 0 && devices[1] == 2);
    for(const char* bad : {"", "0,", ",0", "0,,1", "a", "0,1,2,3"}){
        count = 0;
        assert(rejects([bad, &devices, &count] { cp_cli_devices(bad, devices, count, 3); }));
    }
    count = 0;
    assert(message([&devices, &count] { cp_cli_devices("0,,1", devices, count, 3); }) ==
           "empty device id in \"0,,1\"");
    return 0;
}
