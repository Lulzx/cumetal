#include <cub/cub.h>
#include <cstdio>
#include <cmath>
#include <functional>

static int g_fail = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); g_fail++; } \
    else { printf("PASS: %s\n", msg); } \
} while(0)

static void test_device_partition_if() {
    int input[] = {1, 2, 3, 4, 5, 6, 7, 8};
    int output[8] = {};
    int num_selected = 0;
    size_t temp_bytes = 0;

    // Partition: evens first, then odds
    auto is_even = [](int x) { return x % 2 == 0; };

    cub::DevicePartition::If(nullptr, temp_bytes, input, output, &num_selected, 8, is_even);
    CHECK(temp_bytes > 0, "DevicePartition::If temp_bytes > 0");

    char temp[1];
    cub::DevicePartition::If(temp, temp_bytes, input, output, &num_selected, 8, is_even);
    CHECK(num_selected == 4, "DevicePartition::If selected 4 evens");
    // Evens: 2,4,6,8 then odds: 1,3,5,7
    CHECK(output[0] == 2, "partition evens[0]=2");
    CHECK(output[1] == 4, "partition evens[1]=4");
    CHECK(output[2] == 6, "partition evens[2]=6");
    CHECK(output[3] == 8, "partition evens[3]=8");
}

static void test_device_partition_flagged() {
    int input[] = {10, 20, 30, 40, 50};
    int flags[] = {1, 0, 1, 0, 1};
    int output[5] = {};
    int num_selected = 0;
    size_t temp_bytes = 0;

    cub::DevicePartition::Flagged(nullptr, temp_bytes, input, flags, output, &num_selected, 5);
    char temp[1];
    cub::DevicePartition::Flagged(temp, temp_bytes, input, flags, output, &num_selected, 5);
    CHECK(num_selected == 3, "DevicePartition::Flagged selected 3");
    CHECK(output[0] == 10 && output[1] == 30 && output[2] == 50, "flagged partition correct");
}

static void test_device_merge_sort_keys() {
    int keys[] = {5, 3, 8, 1, 4};
    size_t temp_bytes = 0;

    cub::DeviceMergeSort::SortKeys(nullptr, temp_bytes, keys, 5, std::less<int>());
    char temp[1];
    cub::DeviceMergeSort::SortKeys(temp, temp_bytes, keys, 5, std::less<int>());
    CHECK(keys[0] == 1, "MergeSort keys[0]=1");
    CHECK(keys[1] == 3, "MergeSort keys[1]=3");
    CHECK(keys[2] == 4, "MergeSort keys[2]=4");
    CHECK(keys[3] == 5, "MergeSort keys[3]=5");
    CHECK(keys[4] == 8, "MergeSort keys[4]=8");
}

static void test_device_merge_sort_pairs() {
    int keys[] = {3, 1, 2};
    float vals[] = {30.0f, 10.0f, 20.0f};
    size_t temp_bytes = 0;

    cub::DeviceMergeSort::SortPairs(nullptr, temp_bytes, keys, vals, 3, std::less<int>());
    char temp[1];
    cub::DeviceMergeSort::SortPairs(temp, temp_bytes, keys, vals, 3, std::less<int>());
    CHECK(keys[0] == 1 && keys[1] == 2 && keys[2] == 3, "MergeSort pairs keys sorted");
    CHECK(std::fabs(vals[0] - 10.0f) < 1e-5f, "MergeSort pairs vals[0]=10");
    CHECK(std::fabs(vals[1] - 20.0f) < 1e-5f, "MergeSort pairs vals[1]=20");
    CHECK(std::fabs(vals[2] - 30.0f) < 1e-5f, "MergeSort pairs vals[2]=30");
}

static void test_device_merge_sort_keys_copy() {
    int input[] = {5, 2, 8, 1};
    int output[4] = {};
    size_t temp_bytes = 0;

    cub::DeviceMergeSort::SortKeysCopy(nullptr, temp_bytes, input, output, 4, std::less<int>());
    char temp[1];
    cub::DeviceMergeSort::SortKeysCopy(temp, temp_bytes, input, output, 4, std::less<int>());
    CHECK(output[0] == 1 && output[1] == 2 && output[2] == 5 && output[3] == 8,
          "MergeSort SortKeysCopy correct");
    CHECK(input[0] == 5, "SortKeysCopy input unchanged");
}

int main() {
    test_device_partition_if();
    test_device_partition_flagged();
    test_device_merge_sort_keys();
    test_device_merge_sort_pairs();
    test_device_merge_sort_keys_copy();
    printf("\n%s (%d failures)\n", g_fail ? "SOME TESTS FAILED" : "ALL TESTS PASSED", g_fail);
    return g_fail ? 1 : 0;
}
