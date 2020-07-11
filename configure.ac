AC_PREREQ(2.52)

AC_INIT([apertium-keyboard], [0.1.0], [ftyers@prompsit.com])
AM_INIT_AUTOMAKE

AC_PATH_PROG([GRADLE], [gradle], [false], [$PATH$PATH_SEPARATOR$with_gradle/bin])
AS_IF([test x$GRADLE = xfalse], [AC_MSG_ERROR([You don't have gradle installed.])])

AC_OUTPUT([Makefile apertium-keyboard.pc])

