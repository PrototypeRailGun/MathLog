# -*- coding: utf-8 -*-
__all__ = ["Expression"]

from typing import Union, List
from copy import deepcopy
from MathLog.boolean.code import VALID_SYMBOLS, STN, NTP
from MathLog.boolean.truth_table import TruthTable
from MathLog import sets_of_variables


# Logical operations
operations = {
    "NOT": lambda a: int(not a),
    "AND": lambda a, b: a * b,
    "OR": lambda a, b: min(a + b, 1),
    "XOR": lambda a, b: (a + b) % 2,
    "IMPL": lambda a, b: min((not a) + b, 1),
    "EQ": lambda a, b: int(a == b),
    "NOR": lambda a, b: int(not(a + b)),
    "NAND": lambda a, b: int(not(a * b)),
}


class ExpressionFormatError(Exception):
    def __init__(self, message):
        self.message = message


class Expression:
    """Class for logical expression.

    Here you can calculate the value of an expression with a given set of variables,
    build a truth table, find the DNF, KNF, and Zhegalkin polynomial.
    The class supports magic methods whose operators correspond to conjunction, disjunction, and addition modulo 2.

    """
    def __init__(self, expr, **kwargs):
        """
        :param expr: the logical expression;
        :type expr: str.
        :param kwargs: variables;
        :type kwargs: var=val, где var - name of variable,
            val - value (0, 1, other expression <Expression>)

        """
        self.expr = expr
        if len(self.expr.replace(" ", "").replace("\n", "")) == 0:
            raise ExpressionFormatError("The expression cannot be empty")

        if self.is_correct(expr):
            self._tokens = []
            self.variables = {}
            self._preprocessing()

            for var, val in kwargs.items():
                self.variables.update({var: val})

            self._zhegalkin = None        # Zhegalkin polynomial
            self._dnf = None
            self._cnf = None
            self._truth_table = None
            self._dummy_variables = None
            self._mdnf = None
            self._mcnf = None

    def _preprocessing(self):
        self.expr = self.expr.upper()
        for i in range(len(self.expr)):
            if self.expr[i] == "V":
                if self.expr[i-1] == " " and self.expr[i+1] == " ":
                    self.expr = self.expr[:i] + "+" + self.expr[i+1:]

        i = 0
        while i < len(self.expr):
            if self.expr[i] in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                var = ""
                while i < len(self.expr) and self.expr[i] in \
                        "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                    var += self.expr[i]
                    i += 1

                if var not in self.variables.keys() and var.upper() not in operations.keys():
                    self.expr = self.expr[:i-len(var)] + var.lower() + self.expr[i:]
                    self.variables.update({var.lower(): 0})
            i += 1

        self.expr = self.expr.replace("\n", " ")
        self.expr = self.expr.replace("NOR", "↓")
        self.expr = self.expr.replace("NAND", "|")
        self.expr = self.expr.replace("XOR", "⊕")
        self.expr = self.expr.replace("AND", "*")
        self.expr = self.expr.replace("OR", "+")
        self.expr = self.expr.replace("NOT", "~")
        self.expr = self.expr.replace("EQ", "⇔")
        self.expr = self.expr.replace("IMPL", "→")

    @classmethod
    def _check_brackets(cls, expr):
        """Checking the correctness of the bracket sequence"""
        if expr.count("(") != expr.count(")"):
            raise ExpressionFormatError("The bracket sequence is set incorrectly")
        return True

    @classmethod
    def _check_symbols(cls, expr):
        """Checking for invalid characters"""
        for s in expr:
            if s not in VALID_SYMBOLS:
                raise ExpressionFormatError("Invalid characters are present in the expression: '{}'".format(s))
        return True

    @classmethod
    def is_correct(cls, expr):
        """Checking the correctness of the entire expression"""
        cls._check_brackets(expr)
        cls._check_symbols(expr)
        return True

    def set_variables(self, **kwargs):
        """Adding variable values. Format: <name>=<value>"""
        for var, val in kwargs.items():
            self.variables.update({var: val})

    def __boolean_vectors(self):
        """Boolean vectors of sets of variables"""
        if self._truth_table is None:
            return sets_of_variables(len(self.variables))
        else:
            return list(
                map(lambda vector: [int(v) for v in vector[:len(self.variables)]], self._truth_table.table)
            )

    def _parse(self):
        """Expression parsing

        The source expression (1 + 0) & (~(0+0)&1) IMPL 0 OR (0 * 1) will take the form:
        [['1', 'OR', '0'], 'AND', ['NOT', ['0', 'OR', '0'], 'AND', '1'], 'IMPL', '0', 'OR', ['0', 'AND', '1']]
        This is a tree for recursive descent

        """
        def tokenizer(expr):
            i = 0
            while i < len(self.expr):
                s = expr[i]
                if s in "*&^+∨v~→⇒⊕⇔↓|":
                    yield NTP[STN[s]]
                elif self.expr[i] in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz":
                    var = ""
                    while i < len(self.expr) and self.expr[i] in \
                            "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz":
                        var += self.expr[i]
                        i += 1
                    i -= 1
                    yield var
                elif s in "01()":
                    yield s
                i += 1

        tokens = []
        current = tokens
        brackets = 0
        for t in tokenizer(self.expr):
            if t == "(":
                brackets += 1
                current.append([])
                current = current[-1]
            elif t == ")":
                brackets -= 1
            else:
                current = tokens
                for _ in range(brackets):
                    current = current[-1]
                current.append(t)

        self._tokens = tokens

    def _calculate(self, tokens=None, init=False):
        """Expression evaluation

        :param tokens: tokenized expression
        :param init: is the launch the main one (from the calculate function)
        :type init: bool.
        :return: 0 or 1;
        :rtype: int.

        """
        def delete_dup_not(expression):
            if expression[i+1] == "NOT":
                expression.pop(i)
                expression.pop(i)

        def exec_binary(op, tokens_):
            ind = 0
            while op in tokens_[ind:]:
                ind = tokens_.index(op)
                first = tokens_.pop(ind-1)
                tokens_.pop(ind-1)
                second = tokens_.pop(ind-1)

                tokens_.insert(ind-1, operations[op](
                    self._calculate(tokens=first),
                    self._calculate(tokens=second))
                )

            return tokens_

        def to_return(tokens_):
            if isinstance(tokens_, list):
                tokens_ = tokens_[0]
            if isinstance(tokens_, str):
                if tokens_ == "0" or tokens_ == "1":
                    return int(tokens_)
                else:
                    if isinstance(self.variables[tokens_], int):
                        return self.variables[tokens_]
                    subexpr = self.variables[tokens_]
                    return subexpr.calculate()
            elif isinstance(tokens_, int):
                return tokens_

        if init:
            tokens = deepcopy(self._tokens)

        if isinstance(tokens, int):
            return tokens

        i = 0
        while "NOT" in tokens[i:]:
            i = tokens.index("NOT")
            delete_dup_not(tokens)

            if len(tokens) == 1 and isinstance(tokens, list):
                tokens = tokens[0]
            if len(tokens) > 1:
                tokens.pop(i)
                expr = tokens.pop(i)
                tokens.insert(i, operations["NOT"](self._calculate(tokens=expr)))

        tokens = exec_binary("AND", tokens)
        tokens = exec_binary("NAND", tokens)
        tokens = exec_binary("OR", tokens)
        tokens = exec_binary("XOR", tokens)
        tokens = exec_binary("NOR", tokens)
        tokens = exec_binary("IMPL", tokens)
        tokens = exec_binary("EQ", tokens)

        return to_return(tokens)

    def calculate(self):
        """The interface function for calculating an expression"""
        try:
            if self._truth_table is not None:
                n = int("".join([str(i) for i in self.variables.values()]), base=2)
                return int(self._truth_table.table[n][-1])
            self._parse()
            return self._calculate(init=True)

        except Exception as e:
            raise ValueError(
                """The expression is set incorrectly, or the values of variables are unknown, 
                an error has been intercepted: {}
                The correct expression does not contain parentheses - ((1+0))
                and can only consist of characters
                01()*&^+∨v~→⇒⊕⇔↓|ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz""".format(e)
            )

    def create_truth_table(self, *args):
        """Expression`s truth table

        :param args: additional columns - sub-expressions
        :type args: str
        :return: truth_Table
        :rtype: TruthTable.

        """
        old_vars = deepcopy(self.variables)
        self._truth_table = TruthTable(self, {arg: Expression(arg) for arg in args})
        self.variables = old_vars
        return self._truth_table

    @property
    def truth_table(self):
        if self._truth_table is not None:
            return self._truth_table
        return self.create_truth_table()

    def is_minterm(self):
        """Is a minterm a MathLog expression"""
        if self.variables == {}:
            return False
        old_vars = deepcopy(self.variables)
        variables = list(self.variables)

        count = 0
        for comb in self.__boolean_vectors():
            for i in range(len(variables)):
                self.variables.update(
                    {variables[i]: comb[i]}
                )
            if self.calculate() == 1:
                count += 1
            if count > 1:
                self.variables = old_vars
                return False
        self.variables = old_vars
        return True

    def is_maxterm(self):
        """Is a maxterm a MathLog expression"""
        if self.variables == {}:
            return False
        old_vars = deepcopy(self.variables)
        variables = list(self.variables)

        count = 0
        for comb in self.__boolean_vectors():
            for i in range(len(variables)):
                self.variables.update(
                    {variables[i]: comb[i]}
                )
            if self.calculate() == 0:
                count += 1
            if count > 1:
                self.variables = old_vars
                return False
        self.variables = old_vars
        return True

    def __find_xnf(self, term: int):
        """Finding DNF or CNF"""
        if self.variables == {}:
            return ""
        if term == 1:
            s1, s2 = "*", "+"
        else:
            s1, s2 = "+", "*"

        old_vars = deepcopy(self.variables)
        variables = list(self.variables)

        form = ""
        results = set()
        for comb in self.__boolean_vectors():
            for i in range(len(variables)):
                self.variables.update(
                    {variables[i]: comb[i]}
                )
            r = self.calculate()
            results.add(r)
            if r == term:
                fragment = ""
                for i, val in enumerate(comb):
                    if val == term:
                        fragment += variables[i] + " {} ".format(s1)
                    else:
                        fragment += "~" + variables[i] + " {} ".format(s1)
                form += "(" + fragment[:-3] + ") {} ".format(s2)

        if len(results) == 1:
            return ""
        self.variables = old_vars
        return form[:-3]

    def __find_zhegalkin(self):
        """Finding Zhegalkin polynomial"""
        old_vars = deepcopy(self.variables)
        variables = list(self.variables)
        n = len(variables)
        table = [[] for _ in range(n*2+1)]

        for comb in self.__boolean_vectors():
            for i in range(n):
                self.variables.update(
                    {variables[i]: comb[i]}
                )
                table[i].append(comb[i])
            table[n].append(self.calculate())

        for p in range(n):
            step = 2**p
            i = 0
            xor = False
            while i < 2**n:
                for j in range(step):
                    if xor:
                        table[n+p+1].append(int(table[n+p+1][i-step] != table[n+p][i]))
                    else:
                        table[n+p+1].append(table[n+p][i])
                    i += 1

                xor = not xor

        conjunctions = []
        for i in range(2**n):
            if table[-1][i] == 1:
                con = "*".join(
                    [variables[v] for v in range(n) if table[v][i] == 1]
                )
                if con == "":
                    con = "1"
                conjunctions.append(con)

        self.variables = old_vars
        return " ⊕ ".join(conjunctions)

    def __find_mxnf(self, val: str):
        """Minimizing an expression using the Quine McCluskey method"""
        def mask_to_expr(implicants_):
            """Forming an expression from implicants"""
            if val == "1":
                s1, s2 = "*", "+"
            else:
                s1, s2 = "+", "*"
            form = ""
            for impl_ in implicants_:
                fragment = ""
                for idx, s in enumerate(impl_):
                    if s == val:
                        fragment += variables[idx] + " {} ".format(s1)
                    elif s == "_":
                        pass
                    else:
                        fragment += "~" + variables[idx] + " {} ".format(s1)
                form += "(" + fragment[:-3] + ") {} ".format(s2)
            return form[:-3]

        th = list(self.truth_table.base_row_vectors)
        variables = list(self.variables)
        group_by_count = [[] for _ in range(len(variables)+1)]
        for vector in th:
            if vector[-1] == val:
                group_by_count[vector[:-1].count(val)].append(vector[:-1])

        # Group by position
        by_position = [[] for _ in range(len(variables))]
        for i in range(len(self.variables)):
            for first in group_by_count[i]:
                for second in group_by_count[i+1]:
                    pos = count = 0
                    for p in range(len(variables)):
                        if count > 1:
                            break
                        if first[p] != second[p]:
                            pos = p
                            count += 1
                    else:
                        by_position[pos].append(first[:pos] + "_" + first[pos+1:])

        # Splicing in the groups
        for group in by_position:
            completed = False
            while not completed:
                for first in group:
                    for second in group:
                        pos = count = 0
                        for p in range(len(variables)):
                            if first[p] != second[p]:
                                pos = p
                                count += 1
                            if count > 1:
                                completed = True
                                break
                        else:
                            if first != second:
                                group.remove(first)
                                group.remove(second)
                                group.append(first[:pos] + "_" + first[pos+1:])
                            else:
                                completed = True

        # Absorption
        implicants = []
        for count_group in group_by_count:
            for vector in count_group:
                is_spliced = False
                for group in by_position:
                    for impl in group:
                        if impl not in implicants:
                            implicants.append(impl)
                        for p in range(len(variables)):
                            if impl[p] != "_" and vector[p] != impl[p]:
                                break
                        else:
                            is_spliced = True
                if not is_spliced:
                    implicants.append(vector)

        # Solution of the coverage table
        rows = [vector[:-1] for vector in th if vector[-1] == val]
        table = [[] for _ in range(len(rows))]
        for i, row in enumerate(rows):
            for impl in implicants:
                for p in range(len(variables)):
                    if impl[p] != "_" and impl[p] != row[p]:
                        table[i].append(0)
                        break
                else:
                    table[i].append(1)

        core, new_table, removed_implicants = [], [], []
        for row in table:
            if row.count(1) == 1:
                impl = implicants[row.index(1)]
                if impl not in removed_implicants:
                    core.append(impl)
                    removed_implicants.append(impl)
                for other_row in table:
                    if other_row[row.index(1)] == 1:
                        if other_row in new_table:
                            new_table.remove(other_row)
            elif row not in new_table:
                new_table.append(row)
        table = new_table
        implicants = [i for i in implicants if i not in removed_implicants]

        if not table:
            return mask_to_expr(core)

        combinations = {}
        for comb in sets_of_variables(len(implicants)):
            covered = []
            txnf = []
            for i, v in enumerate(comb):
                if v == 1:
                    txnf.append(implicants[i])
                    for j, row in enumerate(table):
                        if row[i] == 1 and j not in covered:
                            covered.append(j)
            if len(covered) == len(table):
                combinations.update({len(txnf): txnf})
        return mask_to_expr(core + (combinations[sorted(combinations)[0]] if combinations else []))

    @property
    def dnf(self):
        if self._dnf is not None:
            return self._dnf
        dnf = self.__find_xnf(1)
        self._dnf = dnf
        return dnf

    @dnf.setter
    def dnf(self, dnf: str):
        if isinstance(dnf, str):
            self._dnf = dnf
        else:
            raise ExpressionFormatError("The expression must be a string")

    @dnf.deleter
    def dnf(self):
        self._dnf = None

    @property
    def cnf(self):
        if self._cnf is not None:
            return self._cnf
        knf = self.__find_xnf(0)
        self._cnf = knf
        return knf

    @cnf.setter
    def cnf(self, knf: str):
        if isinstance(knf, str):
            self._cnf = knf
        else:
            raise ExpressionFormatError("The expression must be a string")

    @cnf.deleter
    def cnf(self):
        self._cnf = None

    @property
    def zhegalkin(self):
        if self._zhegalkin is not None:
            return self._zhegalkin
        anf = self.__find_zhegalkin()
        return anf

    @zhegalkin.setter
    def zhegalkin(self, zhegalkin: str):
        if isinstance(zhegalkin, str):
            self._zhegalkin = zhegalkin
        else:
            raise ExpressionFormatError("The expression must be a string")

    @zhegalkin.deleter
    def zhegalkin(self):
        self._zhegalkin = None

    @property
    def mdnf(self):
        if self._mdnf is not None:
            return self._mdnf
        mdnf = self.__find_mxnf("1")
        self._mdnf = mdnf
        return mdnf

    @mdnf.setter
    def mdnf(self, mdnf: str):
        if isinstance(mdnf, str):
            self._mdnf = mdnf
        else:
            raise ExpressionFormatError("The expression must be a string")

    @mdnf.deleter
    def mdnf(self):
        self._mdnf = None

    @property
    def mcnf(self):
        if self._mcnf is not None:
            return self._mcnf
        mcnf = self.__find_mxnf("0")
        self._mcnf = mcnf
        return mcnf

    @mcnf.setter
    def mcnf(self, mcnf: str):
        if isinstance(mcnf, str):
            self._mcnf = mcnf
        else:
            raise ExpressionFormatError("The expression must be a string")

    @mcnf.deleter
    def mcnf(self):
        self._mcnf = None

    @property
    def dummy_variables(self):
        if self._dummy_variables is not None:
            return self.dummy_variables

        th = self.create_truth_table()
        values = th.values
        if values.count("1") % 2 != 0:
            return []

        n = len(self.variables)
        step = 2**(n-1)
        dummy = []
        for i in range(n):
            for j in range(0, 2**n-step, 2*step):
                if values[j:j+step] != values[j+step:j+step+step]:
                    break
            else:
                dummy.append(list(self.variables)[i])
            step //= 2
        return dummy

    @dummy_variables.setter
    def dummy_variables(self, variables: List[str]):
        if not isinstance(variables, list):
            raise ValueError("List of dummy variables must have type List[str]")
        self._dummy_variables = variables

    @dummy_variables.deleter
    def dummy_variables(self):
        self._dummy_variables = None

    @classmethod
    def __second_operand(cls, other: Union[str, int, 'Expression']) -> 'Expression':
        if isinstance(other, str):
            second = Expression(other)
        elif isinstance(other, int):
            second = Expression(str(other))
        elif isinstance(other, Expression):
            second = other
        else:
            raise TypeError(
                "parameter other must have type str, int or Expression, not {}".format(type(other))
            )
        return second

    def __and__(self, other: Union[str, int, 'Expression']) -> 'Expression':
        """Conjunctions of two Expressions"""
        second = self.__second_operand(other)
        second._parse()
        self._parse()
        new = "("+self.expr+")" if len(self._tokens) > 1 else self.expr
        new += " * "
        new += "("+second.expr+")" if len(second._tokens) > 1 else second.expr
        return Expression(new)

    def __mul__(self, other: Union[str, int, 'Expression']) -> 'Expression':
        """Conjunctions of two Expressions"""
        return self.__and__(other)

    def __rand__(self, other: Union[str, int, 'Expression']) -> 'Expression':
        """Conjunctions of two Expressions"""
        second = self.__second_operand(other)
        second._parse()
        self._parse()
        new = "(" + second.expr + ")" if len(second._tokens) > 1 else second.expr
        new += " * "
        new += "(" + self.expr + ")" if len(self._tokens) > 1 else self.expr
        return Expression(new)

    def __rmul__(self, other: Union[str, int, 'Expression']) -> 'Expression':
        """Conjunctions of two Expressions"""
        return self.__rand__(other)

    def __or__(self, other: Union[str, int, 'Expression']) -> 'Expression':
        """Disjunctions of two Expressions"""
        second = self.__second_operand(other)
        second._parse()
        self._parse()
        new = "("+self.expr+")" if len(self._tokens) > 1 else self.expr
        new += " + "
        new += "("+second.expr+")" if len(second._tokens) > 1 else second.expr
        return Expression(new)

    def __add__(self, other: Union[str, int, 'Expression']) -> 'Expression':
        """Disjunctions of two Expressions"""
        return self.__or__(other)

    def __ror__(self, other: Union[str, int, 'Expression']) -> 'Expression':
        """Disjunctions of two Expressions"""
        second = self.__second_operand(other)
        second._parse()
        self._parse()
        new = "(" + second.expr + ")" if len(second._tokens) > 1 else second.expr
        new += " + "
        new += "(" + self.expr + ")" if len(self._tokens) > 1 else self.expr
        return Expression(new)

    def __radd__(self, other: Union[str, int, 'Expression']) -> 'Expression':
        """Disjunctions of two Expressions"""
        return self.__ror__(other)

    def __xor__(self, other: Union[str, int, 'Expression']) -> 'Expression':
        """Additional modulo 2 of two Expressions"""
        second = self.__second_operand(other)
        second._parse()
        self._parse()
        new = "("+self.expr+")" if len(self._tokens) > 1 else self.expr
        new += " ⊕ "
        new += "("+second.expr+")" if len(second._tokens) > 1 else second.expr
        return Expression(new)

    def __rxor__(self, other: Union[str, int, 'Expression']) -> 'Expression':
        """Additional modulo 2 of two Expressions"""
        second = self.__second_operand(other)
        second._parse()
        self._parse()
        new = "(" + second.expr + ")" if len(second._tokens) > 1 else second.expr
        new += " ⊕ "
        new += "(" + self.expr + ")" if len(self._tokens) > 1 else self.expr
        return Expression(new)

    def __eq__(self, other: Union[str, int, 'Expression']) -> bool:
        values1 = self.truth_table.values
        second = self.__second_operand(other)
        values2 = second.truth_table.values
        return values1 == values2

    def __ne__(self, other: Union[str, int, 'Expression']) -> bool:
        return not self.__eq__(other)

    def __str__(self):
        return self.expr


def main():
    """Example of using a class Expression"""
    expr = Expression("~x2 v ((x1 * ~x3) | ~(x2 | ~x3))")
    print("Expression: ", str(expr))
    print("DNF: ", expr.dnf)
    print("CNF: ", expr.cnf)
    print("Zhegalkin polynomial: ", expr.zhegalkin)
    print("Truth table:")
    print(expr.truth_table)


if __name__ == "__main__":
    main()
