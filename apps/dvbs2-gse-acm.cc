/* SPDX-License-Identifier: Apache-2.0 */
// Thin role-based launcher for the production ACM modem over radio_tcp.
#include <limits.h>
#include <unistd.h>
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    char self[PATH_MAX];
    ssize_t n = readlink("/proc/self/exe", self, sizeof(self) - 1);
    if (n <= 0) { perror("readlink"); return 1; }
    self[n] = '\0';
    std::string trx(self);
    auto slash = trx.find_last_of('/');
    trx = trx.substr(0, slash + 1) + "dvbs2-uhd-trx";

    std::vector<std::string> values{trx, "--radio-driver", "tcp", "--acm"};
    for (int i = 1; i < argc; ++i) values.emplace_back(argv[i]);
    std::vector<char*> args;
    for (auto& value : values) args.push_back(value.data());
    args.push_back(nullptr);
    execv(trx.c_str(), args.data());
    perror("execv dvbs2-uhd-trx");
    return 1;
}
