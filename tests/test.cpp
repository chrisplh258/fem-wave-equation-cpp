#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <regex>
#include <string>

#ifndef BIN
#define BIN "./build/wave_solver"
#endif

// Run solver, redirect output to file
int run_solver(const std::string &args, const std::string &outfile)
{
    std::string cmd = std::string(BIN) + " " + args + " > " + outfile + " 2>&1";
    return std::system(cmd.c_str());
}

// Parse L2 error from solver output 
double parse_L2(const std::string &outfile)
{
    std::ifstream in(outfile);
    std::string s((std::istreambuf_iterator<char>(in)),
                  std::istreambuf_iterator<char>());

    std::smatch m;
    // 1) "Final-time L2 error = <num>"
    std::regex re1(R"(Final-time\s+L2\s+error\s*=\s*([-+0-9.eE]+))");
    if (std::regex_search(s, m, re1)) return std::stod(m[1]);

    // 2) "L2 error = <num>"
    std::regex re2(R"(\bL2\s*error\s*=\s*([-+0-9.eE]+))");
    if (std::regex_search(s, m, re2)) return std::stod(m[1]);

    // 3) Optional summary style "L2_error=<num>"
    std::regex re3(R"(\bL2_error=([-+0-9.eE]+))");
    if (std::regex_search(s, m, re3)) return std::stod(m[1]);

    std::cerr << "[parse_L2] Could not find L2 error in " << outfile << "\n--- file contents ---\n"
              << s << "\n----------------------\n";
    return -1.0;
}

int main()
{
    int passed = 0, failed = 0;

    // 1) Baseline accuracy 
    {
        const char *args = "--h 0.04 --dt 2.5e-4 --c 1.0";
        int rc = run_solver(args, "out1.txt");
        if (rc != 0) { std::cout << "Test1 FAIL (rc=" << rc << ")\n"; ++failed; }
        else {
            double L2 = parse_L2("out1.txt");
            if (L2 > 0 && L2 < 2e-2) { std::cout << "Test1 OK (L2=" << L2 << ")\n"; ++passed; }
            else { std::cout << "Test1 FAIL (L2=" << L2 << ")\n"; ++failed; }
        }
    }

    // 2) Refinement improves error 
    {
        int rc1 = run_solver("--h 0.04 --dt 2.5e-4 --c 1.0", "out2a.txt");
        int rc2 = run_solver("--h 0.02 --dt 1.25e-4 --c 1.0", "out2b.txt");
        if (rc1 != 0 || rc2 != 0) {
            std::cout << "Test2 FAIL (rc1=" << rc1 << ", rc2=" << rc2 << ")\n"; ++failed;
        } else {
            double E1 = parse_L2("out2a.txt");
            double E2 = parse_L2("out2b.txt");
            if (E2 > 0 && E2 < E1) {
                std::cout << "Test2 OK (E1=" << E1 << ", E2=" << E2 << ")\n"; ++passed;
            } else {
                std::cout << "Test2 FAIL (E1=" << E1 << ", E2=" << E2 << ")\n"; ++failed;
            }
        }
    }

    // 3) Robustness: invalid h should fail
    {
        int code = run_solver("--h -0.1", "out3.txt");
        if (code != 0) { std::cout << "Test3 OK\n"; ++passed; }
        else { std::cout << "Test3 FAIL\n"; ++failed; }
    }

    // 4) Robustness: invalid dt should fail
    {
        int code = run_solver("--h 0.04 --dt 0", "out4.txt");
        if (code != 0) { std::cout << "Test4 OK\n"; ++passed; }
        else { std::cout << "Test4 FAIL\n"; ++failed; }
    }

    // 5) Performance: small run completes in under 5 seconds
    {
        auto t0 = std::chrono::steady_clock::now();
        int rc = run_solver("--h 0.05 --dt 1e-3 --c 1.0", "out5.txt");
        auto t1 = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        if (rc == 0 && ms < 5000) { std::cout << "Test5 OK (" << ms << " ms)\n"; ++passed; }
        else { std::cout << "Test5 FAIL (rc=" << rc << ", " << ms << " ms)\n"; ++failed; }
    }

    std::cout << "Passed: " << passed << ", Failed: " << failed << "\n";
    return failed == 0 ? 0 : 1;
}
