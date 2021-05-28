# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Some functions related to files and the file system"""
import os
import re
import contextlib
import sys


@contextlib.contextmanager
def chdir(path):
    """Context manager to temporarily change the working directory

    Parameters
    ----------
    path : str
        Path of the directory to change to. :py:func:`os.path.expanduser` is
        called on this.

    Examples
    --------
    >>> with chdir("subdir"):
    ...     # here the working directory is "subdir"
    >>> # here we are back
    """
    old_wd = os.getcwd()
    os.chdir(os.path.expanduser(str(path)))
    try:
        yield
    finally:
        os.chdir(old_wd)


def get_files(pattern, subdir=os.curdir):
    r"""Get all files matching a regular expression

    Parameters
    ----------
    pattern : str
        Regular expression to search in the file name. Search is performed
        on the path relative to `subdir`. One can also define groups (using
        parenthesis), which will be returned in addition to the matching
        file names. **A note to Windows users: Use a forward slash (/) for
        subdirectories.**
    subdir : str or pathlib.Path, optional
        Any regular expression matching will be performed relative to `subdir`.
        Defaults to ``os.curdir``.

    Returns
    -------
    files : list of str
        Sorted list of file where `pattern` could be matched.
    groups : list of tuple
        Values of the groups defined in the `pattern`. Values are converted
        to int if possible, otherwise try converting to float. If that fails
        as well, use the string.

    Examples
    --------
    >>> names, ids = get_files(r"^image_(.*)_(\d{3}).tif$", "subdir")
    >>> names
    ['image_xxx_001.tif', 'image_xxx_002.tif', 'image_yyy_003.tif']
    >>> ids
    [('xxx', 1), ('xxx', 2), ('yyy', 3)]
    """
    r = re.compile(str(pattern))
    flist = []
    idlist = []
    sd = str(subdir)
    for dp, dn, fn in os.walk(sd):
        reldir = os.path.relpath(dp, sd)
        if reldir == os.curdir:
            # Do not prefix "./"
            reldir = ""
        for f in fn:
            relp = os.path.join(reldir, f)
            if sys.platform == "win32":
                relp = relp.replace("\\", "/")
            m = r.search(relp)
            if m is None:
                continue
            flist.append(relp)
            ids = []
            for i in m.groups():
                try:
                    ids.append(int(i))
                except ValueError:
                    try:
                        ids.append(float(i))
                    except ValueError:
                        ids.append(i)
            idlist.append(ids)
    slist = sorted(zip(flist, idlist), key=lambda x: x[0])
    return [s[0] for s in slist], [tuple(s[1]) for s in slist]
