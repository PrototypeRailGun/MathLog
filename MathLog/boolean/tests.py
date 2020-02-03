import unittest
from MathLog import Expression


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.expr1 = Expression("(~(p v q) ^ r) v ~ (q v r)")
        self.expr2 = Expression("1 IMPL (1 | 0) ^ 1")
        self.expr3 = Expression("hello")
        self.expr4 = Expression("hello + WORLD")

    def test_calculate(self):
        self.assertEqual(self.expr1.calculate(), 1)
        self.assertEqual(self.expr2.calculate(), 1)
        self.assertEqual(self.expr3.calculate(), 0)

    def test_dnf(self):
        self.assertEqual(self.expr1.dnf, "(~p * ~q * ~r) + (~p * ~q * r) + (p * ~q * ~r)")
        self.assertEqual(self.expr2.dnf, "")
        self.assertEqual(self.expr3.dnf, "hello")
        self.assertEqual(self.expr4.dnf, "(~hello * world) + (hello * ~world) + (hello * world)")

    def test_cnf(self):
        self.assertEqual(
            self.expr1.cnf, "(p + ~q + r) * (p + ~q + ~r) * (~p + q + ~r) * (~p + ~q + r) * (~p + ~q + ~r)"
        )
        self.assertEqual(self.expr2.cnf, "")
        self.assertEqual(self.expr3.cnf, "hello")
        self.assertEqual(self.expr4.cnf, "(hello + world)")

    def test_zhegalkin(self):
        e1 = Expression("(x1 * x2 + x3) IMPL ~x2")
        self.assertEqual(e1.zhegalkin, "1 ⊕ x2*x3 ⊕ x1*x2 ⊕ x1*x2*x3")
        self.assertEqual(Expression("x v y").zhegalkin, "y ⊕ x ⊕ x*y")

    def test_is_minterm_and_maxterm(self):
        self.assertFalse(self.expr1.is_minterm())
        self.assertFalse(self.expr1.is_maxterm())
        self.assertFalse(self.expr2.is_minterm())
        self.assertFalse(self.expr2.is_maxterm())
        self.assertTrue(self.expr3.is_minterm())
        self.assertTrue(self.expr3.is_maxterm())
        self.assertFalse(self.expr4.is_minterm())
        self.assertTrue(self.expr4.is_maxterm())

    def test_truth_table(self):
        self.assertEqual(
            repr(self.expr1.truth_table),
            str(["0001", "0011", "0100", "0110", "1001", "1010", "1100", "1110"])
        )
        self.assertEqual(repr(self.expr2.truth_table), str(["1"]))
        self.assertEqual(repr(self.expr3.truth_table), str(["00", "11"]))
        self.assertEqual(repr(self.expr4.truth_table), str(["000", "011", "101", "111"]))

        self.assertEqual(
            list(self.expr1.create_truth_table("p v q", "q v r").col_vectors),
            ["00001111", "00110011", "01010101", "00111111", "01110111", "11001000"]
        )
        self.assertEqual(
            list(self.expr1.truth_table.base_row_vectors),
            ["0001", "0011", "0100", "0110", "1001", "1010", "1100", "1110"]
        )
        self.assertEqual(
            list(self.expr1.truth_table.base_col_vectors),
            ["00001111", "00110011", "01010101", "11001000"]
        )

    def test_truth_table_magic(self):
        th1 = self.expr1.create_truth_table("p ^ q")
        th2 = self.expr2.create_truth_table()
        th3 = self.expr3.create_truth_table()
        th4 = self.expr4.create_truth_table()
        th5 = Expression("p + q + r").truth_table

        for i in th1:
            print(i)
        print()
        for i in th4.row_vectors:
            print(i)
        print()
        for i in th4.col_vectors:
            print(i)
        print()
        for i in th4.base_row_vectors:
            print(i)
        print()
        for i in th4.base_col_vectors:
            print(i)

        print(len(th1), len(th2), len(th3), len(th4))   # 8 1 2 4

        self.assertTrue(th1 == th1)
        self.assertFalse(th2 == th3)
        self.assertTrue(th1 > th4)
        self.assertFalse(th2 > th3)
        self.assertTrue(th4 >= th3)
        self.assertFalse(th1 <= th2)

        self.assertEqual(list(reversed(th4)), ["111", "101", "011", "000"])
        self.assertEqual(th1[2], "01000")
        th1[0] = "11111"
        th4[3] = "000"
        print(-th1, ~th1)
        print(th1 | th5)
        print(th1 ^ th5)

        self.assertEqual(repr(th1 & th5), "['00000', '00101', '01000', '01100', '10001', '10100', '11010', '11110']")
        self.assertEqual(th1.values, "11001000")

    def test_expression_magic(self):
        self.assertEqual(str(self.expr4 * "a * b v c"), "(hello + world) * (a * b + c)")
        self.assertEqual(str(self.expr4 * 1), "(hello + world) * 1")
        self.assertEqual(str(self.expr4 & self.expr1), "(hello + world) * ((~(p + q) ^ r) + ~ (q + r))")

        self.assertEqual(str(self.expr4 + "a * b v c"), "(hello + world) + (a * b + c)")
        self.assertEqual(str(self.expr4 + "a"), "(hello + world) + a")
        self.assertEqual(str(self.expr4 | self.expr1), "(hello + world) + ((~(p + q) ^ r) + ~ (q + r))")

        self.assertEqual(str(self.expr4 ^ "a * b v c"), "(hello + world) ⊕ (a * b + c)")
        self.assertEqual(str(self.expr4 ^ "fff"), "(hello + world) ⊕ fff")
        self.assertEqual(str(self.expr4 ^ self.expr1), "(hello + world) ⊕ ((~(p + q) ^ r) + ~ (q + r))")


if __name__ == "__main__":
    unittest.main()
