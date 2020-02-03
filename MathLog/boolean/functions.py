# -*- coding: utf-8 -*-


def minterm_knf(*args):
    """Нахождение КНФ минтерма по набору аргументов, дающих в значении функции 1

    :param args: набор значений переменных
    :type args: int
    :return КНФ
    :rtype: str

    """
    knf_ = ""
    for i, val in enumerate(args):
        if val == 1:
            knf_ += "p{} * ".format(i)
        else:
            knf_ += "~p{} * ".format(i)
    return knf_[:-3]


def maxterm_dnf(*args):
    """Нахождение ДНФ макстерма по набору аргументов, дающих в значении функции 0

    :param args: набор значений переменных,
    :type args: int.
    :return ДНФ;
    :rtype: str.

    """
    dnf_ = ""
    for i, val in enumerate(args):
        if val == 0:
            dnf_ += "p{} * ".format(i)
        else:
            dnf_ += "~p{} * ".format(i)
    return dnf_[:-3]


def dnf_by_sets(*args):
    """Нахождение ДНФ функции по набору ее аргументов, дающих в значении функции 1

    :param args: список наборов значений,
    :type args: List[int].
    :return: ДНФ;
    :rtype: str.

    """
    return "(" + ") + (".join(
        list(
            map(lambda s: minterm_knf(*s), args)
        )
    ) + ")"


def knf_by_sets(*args):
    """Нахождение КНФ функции по набору ее аргументов, дающих в значении функции 0

    :param args: список наборов значений,
    :type args: List[int],
    :return: КНФ,
    :rtype: str

    """
    return "(" + ") * (".join(
        list(
            map(lambda s: maxterm_dnf(*s), args)
        )
    ) + ")"


def sets_of_variables(n: int):
    for set_ in range(2**n):
        yield [int(s) for s in bin(set_)[2:].rjust(n, "0")]
