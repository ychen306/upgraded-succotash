static void handle_solution(int depth) __attribute__((noinline));
static void handle_solution(int depth) {
  for (int i = 0; i <= depth; i++)
    solution_handlers[i](solution[i]);
  printf("\n");
}

static bool memcmp_256bit_eq_u(const void *str1, const void *str2, size_t count)
{
  const __m256i *s1 = (__m256i*)str1;
  const __m256i *s2 = (__m256i*)str2;

  while (count--)
  {
    __m256i item1 = _mm256_lddqu_si256(s1++);
    __m256i item2 = _mm256_lddqu_si256(s2++);
    __m256i result = _mm256_cmpeq_epi64(item1, item2);
    // cmpeq returns 0xFFFFFFFFFFFFFFFF per 64-bit portion where equality is
    // true, and 0 per 64-bit portion where false

    // If result is not all ones, then there is a difference here.
    // This is the same thing as _mm_test_all_ones, but 256-bit
    if(!(unsigned int)_mm256_testc_si256(result, _mm256_set1_epi64x(0xFFFFFFFFFFFFFFFF)))
    { // Using 0xFFFFFFFFFFFFFFFF explicitly instead of -1 for clarity.
      // It really makes no difference on two's complement machines.
      return false;
    }
  }
  return true;
}

static bool fast_memcmp(const void *a, const void *b, int count) {
  return memcmp_256bit_eq_u(a, b, count >> 5);
}

template <typename InstEnumerator>
void enum_insts(int num_tests, const EnumNode *node, int depth) {
  int num_insts = InstEnumerator::num_insts();
  if (InstEnumerator::has_insts()) {
    for (int i = 0; i < num_insts; i++) {
      int div_by_zero = InstEnumerator::run_inst(num_tests, i);
      if (depth == 4)
        num_enumerated += 1;

      if (__builtin_expect(div_by_zero, 0))
        continue;

      solution_handlers[depth] = InstEnumerator::solution_handler;
      solution[depth] = i;
      if (__builtin_expect(InstEnumerator::check(num_tests), 0)) {
        handle_solution(depth);
      }

      int num_children = node->num_children;
      for (int j = 0; j < num_children; j++) {
        auto *child = node->children[j];
        child->enumerate(num_tests, child, depth+1);
      }
    }
  } else {
    int num_children = node->num_children;
    for (int j = 0; j < num_children; j++) {
      auto *child = node->children[j];
      child->enumerate(num_tests, child, depth);
    }
  }
}
