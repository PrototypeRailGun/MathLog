from copy import deepcopy
from typing import Iterable
from MathLog import sets_of_variables


__all__ = ["TruthTable"]


class TruthTable:
    """This class is the truth table of the Boolean expression.

    Attributes:
    columns: List[str] - columns of the truth table
    table: List[str] - a truth table defined by Boolean vectors
    expression: Expression - the expression that this truth table belongs to
    variables: List[str] - variables of this expression

    if you need more information, write 'help(TruthTable)'

    Look for usage examples in the expression, example, and __init__ modules

    """
    def __init__(self, expression, subexpressions: dict) -> None:
        """ Truth table initialisation

        :param expression: MathLog expression
        :type expression: Expression
        :param subexpressions: additional column that are subexpressions
        :type subexpressions: <str>=<Expression>

        """
        if expression.variables == {}:
            self.columns = ["E"]
            self.table = [str(expression.calculate())]
            self.variables = {}
            return

        self.expression = expression
        variables = list(expression.variables.keys())
        self.variables = variables
        self.subexpressions = subexpressions
        n = len(variables)
        self.columns = variables + [subexpr.expr for subexpr in subexpressions.values()] + ["F"]

        table = []
        for comb in sets_of_variables(n):
            row_vector = ""
            for i, v in enumerate(comb):
                row_vector += str(v)
                self.expression.variables.update({variables[i]: v})
                for subexpr in subexpressions.values():
                    if variables[i] in subexpr.variables.keys():
                        subexpr.variables.update({variables[i]: v})

            for subexpr in subexpressions.values():
                row_vector += str(subexpr.calculate())
            row_vector += str(self.expression.calculate())
            table.append(row_vector)
        self.table = table

    @property
    def row_vectors(self) -> Iterable[str]:
        """<Iterator>: MathLog vectors of truth table rows"""
        for row in self.table:
            yield row

    @property
    def base_row_vectors(self) -> Iterable[str]:
        """<Iterator>: MathLog vectors of truth table rows without additional column components"""
        n = len(self.variables)
        for row in self.table:
            yield row[:n] + row[-1]

    @property
    def col_vectors(self) -> Iterable[str]:
        """<Iterator>: MathLog vectors of truth table columns"""
        for col in zip(*self.table):
            yield "".join(col)

    @property
    def base_col_vectors(self) -> Iterable[str]:
        """<Iterator>: MathLog vectors of truth table columns without additional columns"""
        for col in zip(*self.base_row_vectors):
            yield "".join(col)

    @property
    def values(self) -> str:
        """The last column of the truth table - the value column"""
        return "".join(row[-1] for row in self.table)

    def __eq__(self, other: 'TruthTable') -> bool:
        """Truth tables are equal if the numbers of their rows are equal"""
        return len(self.table) == len(other.table)

    def __ne__(self, other: 'TruthTable') -> bool:
        return not self.__eq__(other)

    def __lt__(self, other: 'TruthTable') -> bool:
        """Of the two truth tables, the one with fewer rows is the smaller"""
        return len(self.table) < len(other.table)

    def __gt__(self, other: 'TruthTable') -> bool:
        """Of the two truth tables, the one with the most rows is the largest."""
        return len(self.table) > len(other.table)

    def __le__(self, other: 'TruthTable') -> bool:
        """Of the two truth tables, the one with the most rows is the largest."""
        return len(self.table) <= len(other.table)

    def __ge__(self, other: 'TruthTable') -> bool:
        """Of the two truth tables, the one with the most rows is the largest."""
        return len(self.table) >= len(other.table)

    def __invert__(self) -> 'TruthTable':
        """In last column 0 changes to 1 and 1 changes to 0"""
        for i, row_vector in enumerate(self.table):
            self.table[i] = row_vector[:-1] + str(int(not int(row_vector[-1])))
        return self

    def __and__(self, other: 'TruthTable') -> 'TruthTable':
        """Conjunction of the values of the last columns. Truth tables must be equal (rule as in __eq__)

        :return: new truth table with a new expression (conjunction of expressions self and other)
            and additional columns for tables.
        :rtype: 'TruthTable'
        """
        if not self.__eq__(other):
            raise ValueError("Truth tables are not equal")

        subexpressions = deepcopy(self.subexpressions)
        for name, subexpr in other.subexpressions.items():
            subexpressions.update({name: subexpr})
        return TruthTable(self.expression * other.expression, subexpressions)

    def __mul__(self, other: 'TruthTable') -> 'TruthTable':
        """Conjunction of the values of the last columns. Truth tables must be equal (rule as in __eq__)"""
        return self.__and__(other)

    def __rand__(self, other: 'TruthTable') -> 'TruthTable':
        """Conjunction of the values of the last columns. Truth tables must be equal (rule as in __eq__)"""
        return self.__and__(other)

    def __rmul__(self, other: 'TruthTable') -> 'TruthTable':
        """Conjunction of the values of the last columns. Truth tables must be equal (rule as in __eq__)"""
        return self.__and__(other)

    def __or__(self, other: 'TruthTable'):
        """Disjunction of the values of the last columns. Truth tables must be equal (rule as in __eq__)

        :return: new truth table with a new expression (disjunction of expressions self and other)
            and additional columns for tables.
        :rtype: 'TruthTable'
        """
        if not self.__eq__(other):
            raise ValueError("Truth tables are not equal")

        subexpressions = deepcopy(self.subexpressions)
        for name, subexpr in other.subexpressions.items():
            subexpressions.update({name: subexpr})
        return TruthTable(self.expression | other.expression, subexpressions)

    def __add__(self, other: 'TruthTable') -> 'TruthTable':
        """Disjunction of the values of the last columns. Truth tables must be equal (rule as in __eq__)"""
        return self.__or__(other)

    def __ror__(self, other: 'TruthTable') -> 'TruthTable':
        """Disjunction of the values of the last columns. Truth tables must be equal (rule as in __eq__)"""
        return self.__or__(other)

    def __radd__(self, other: 'TruthTable') -> 'TruthTable':
        """Disjunction of the values of the last columns. Truth tables must be equal (rule as in __eq__)"""
        return self.__or__(other)

    def __xor__(self, other: 'TruthTable') -> 'TruthTable':
        """Addition modulo 2 of the values of the last columns. Truth tables must be equal (rule as in __eq__)

        :return: new truth table with a new expression (addition modulo 2 of expressions self and other)
            and additional columns for tables.
        :rtype: 'TruthTable'
        """
        if not self.__eq__(other):
            raise ValueError("Truth tables are not equal")

        subexpressions = deepcopy(self.subexpressions)
        for name, subexpr in other.subexpressions.items():
            subexpressions.update({name: subexpr})
        return TruthTable(self.expression ^ other.expression, subexpressions)

    def __rxor__(self, other: 'TruthTable') -> 'TruthTable':
        """Addition modulo 2 of the values of the last columns. Truth tables must be equal (rule as in __eq__)"""
        return self.__xor__(other)

    def __neg__(self) -> 'TruthTable':
        """In last column 0 changes to 1 and 1 changes to 0"""
        return self.__invert__()

    def __len__(self) -> int:
        """The numbers of rows in truth table"""
        return len(self.table)

    def __getitem__(self, key: int) -> str:
        """Getting truth table row"""
        if not isinstance(key, int):
            raise TypeError("Parameter 'key' must by an integer")
        if 0 > key or key > len(self.table):
            raise IndexError()
        return self.table[key]

    def __setitem__(self, key: int, value: str) -> None:
        """Setting truth table row"""
        if not isinstance(key, int):
            raise TypeError("Parameter 'key' must by an integer")
        if 0 > key or key > len(self.table):
            raise IndexError()
        if not isinstance(value, str):
            raise ValueError("Boolean vector must be a string")
        if len(value) != len(self.columns):
            raise ValueError(
                "The new MathLog vector must have as many components as there are columns in the truth table"
            )
        self.table[key] = value

    def __delitem__(self, key: int) -> None:
        """Deleting truth_table row"""
        if not isinstance(key, int):
            raise TypeError("Parameter 'key' must by an integer")
        if 0 > key or key > len(self.table):
            raise IndexError()
        del self.table[key]

    def __iter__(self) -> Iterable[str]:
        """Row vectors - TruthTable.row_vectors"""
        return self.row_vectors

    def __reversed__(self) -> Iterable[str]:
        """reversed(<list of row vectors>) """
        return reversed(self.table)

    def __contains__(self, item: str) -> bool:
        """Is there such a string in the truth table"""
        return item in self.table

    def __call__(self, *args) -> None:
        """Call __init__ with new arguments set"""
        self.__init__(*args)

    def __str__(self) -> str:
        """Truth table as a table"""
        space = [len(col) for col in self.columns]
        view = " | ".join(self.columns) + "\n"
        for row_vector in self.table:
            line = ""
            for i in range(len(self.columns)):
                if space[i] % 2 == 1:
                    line += " "*(space[i]//2) + str(row_vector[i]) + " "*(space[i]//2) + "   "
                else:
                    line += " "*(space[i]//2-1) + str(row_vector[i]) + " "*(space[i]//2) + "   "
            view += line + "\n"
        return view

    def __repr__(self) -> str:
        return str(self.table)

    def __bool__(self):
        return bool(self.table)
