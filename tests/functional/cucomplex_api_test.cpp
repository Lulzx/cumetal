#include <cuComplex.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>

static int g_fail = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); g_fail++; } \
    else { printf("PASS: %s\n", msg); } \
} while(0)

#define APPROX(a, b) (fabs((a) - (b)) < 1e-5)

static void test_make_and_accessors() {
    cuFloatComplex c = make_cuFloatComplex(3.0f, 4.0f);
    CHECK(cuCrealf(c) == 3.0f && cuCimagf(c) == 4.0f, "make_cuFloatComplex + accessors");

    cuDoubleComplex d = make_cuDoubleComplex(1.0, 2.0);
    CHECK(cuCreal(d) == 1.0 && cuCimag(d) == 2.0, "make_cuDoubleComplex + accessors");

    cuComplex alias = make_cuComplex(5.0f, 6.0f);
    CHECK(cuCrealf(alias) == 5.0f, "cuComplex alias");
}

static void test_arithmetic_float() {
    cuFloatComplex a = make_cuFloatComplex(1.0f, 2.0f);
    cuFloatComplex b = make_cuFloatComplex(3.0f, 4.0f);

    cuFloatComplex s = cuCaddf(a, b);
    CHECK(s.x == 4.0f && s.y == 6.0f, "cuCaddf");

    cuFloatComplex d = cuCsubf(a, b);
    CHECK(d.x == -2.0f && d.y == -2.0f, "cuCsubf");

    // (1+2i)(3+4i) = 3+4i+6i+8i^2 = -5+10i
    cuFloatComplex m = cuCmulf(a, b);
    CHECK(APPROX(m.x, -5.0f) && APPROX(m.y, 10.0f), "cuCmulf");

    // (1+2i)/(3+4i) = (1+2i)(3-4i)/25 = (11+2i)/25
    cuFloatComplex q = cuCdivf(a, b);
    CHECK(APPROX(q.x, 11.0f/25.0f) && APPROX(q.y, 2.0f/25.0f), "cuCdivf");
}

static void test_arithmetic_double() {
    cuDoubleComplex a = make_cuDoubleComplex(1.0, 2.0);
    cuDoubleComplex b = make_cuDoubleComplex(3.0, 4.0);

    cuDoubleComplex s = cuCadd(a, b);
    CHECK(s.x == 4.0 && s.y == 6.0, "cuCadd");

    cuDoubleComplex m = cuCmul(a, b);
    CHECK(APPROX(m.x, -5.0) && APPROX(m.y, 10.0), "cuCmul");
}

static void test_abs_conj_neg() {
    cuFloatComplex c = make_cuFloatComplex(3.0f, 4.0f);
    CHECK(APPROX(cuCabsf(c), 5.0f), "cuCabsf");

    cuFloatComplex conj = cuConjf(c);
    CHECK(conj.x == 3.0f && conj.y == -4.0f, "cuConjf");

    cuFloatComplex neg = cuCnegf(c);
    CHECK(neg.x == -3.0f && neg.y == -4.0f, "cuCnegf");

    cuDoubleComplex d = make_cuDoubleComplex(3.0, 4.0);
    CHECK(APPROX(cuCabs(d), 5.0), "cuCabs");
}

static void test_conversion() {
    cuFloatComplex f = make_cuFloatComplex(1.5f, 2.5f);
    cuDoubleComplex d = cuComplexFloatToDouble(f);
    CHECK(d.x == 1.5 && d.y == 2.5, "cuComplexFloatToDouble");

    cuFloatComplex f2 = cuComplexDoubleToFloat(d);
    CHECK(f2.x == 1.5f && f2.y == 2.5f, "cuComplexDoubleToFloat");
}

int main() {
    test_make_and_accessors();
    test_arithmetic_float();
    test_arithmetic_double();
    test_abs_conj_neg();
    test_conversion();

    printf("\n%s (%d failures)\n", g_fail ? "SOME TESTS FAILED" : "ALL TESTS PASSED", g_fail);
    return g_fail ? 1 : 0;
}
