"""Generic persistent, concurrent dictionary-like facility."""

from __future__ import division, with_statement, absolute_import

__copyright__ = """
Copyright (C) 2011,2014 Andreas Kloeckner
Copyright (C) 2017 Matt Wala
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import logging

try:
    import collections.abc as abc
except ImportError:
    # Python 2
    import collections as abc

import os
import shutil
import sys
import errno

import six

logger = logging.getLogger(__name__)

__doc__ = """
Persistent Hashing and Persistent Dictionaries
==============================================

This module contains functionality that allows hashing with keys that remain
valid across interpreter invocations, unlike Python's built-in hashes.

This module also provides a disk-backed dictionary that uses persistent hashing.

.. autoexception:: NoSuchEntryError
.. autoexception:: ReadOnlyEntryError

.. autoexception:: CollisionWarning

.. autoclass:: KeyBuilder
.. autoclass:: PersistentDict
.. autoclass:: WriteOncePersistentDict
"""

try:
    import hashlib
    new_hash = hashlib.sha256
except ImportError:
    # for Python << 2.5
    import sha
    new_hash = sha.new


def _make_dir_recursively(dir_):
    try:
        os.makedirs(dir_)
    except OSError as e:
        from errno import EEXIST
        if e.errno != EEXIST:
            raise


def update_checksum(checksum, obj):
    if isinstance(obj, six.text_type):
        checksum.update(obj.encode("utf8"))
    else:
        checksum.update(obj)


# {{{ cleanup managers

class CleanupBase(object):
    pass


class CleanupManager(CleanupBase):
    def __init__(self):
        self.cleanups = []

    def register(self, c):
        self.cleanups.insert(0, c)

    def clean_up(self):
        for c in self.cleanups:
            c.clean_up()

    def error_clean_up(self):
        for c in self.cleanups:
            c.error_clean_up()


class LockManager(CleanupBase):
    def __init__(self, cleanup_m, lock_file, stacklevel=0):
        self.lock_file = lock_file

        attempts = 0
        while True:
            try:
                self.fd = os.open(self.lock_file,
                        os.O_CREAT | os.O_WRONLY | os.O_EXCL)
                break
            except OSError:
                pass

            from time import sleep
            sleep(1)

            attempts += 1

            if attempts > 10:
                from warnings import warn
                warn("could not obtain lock--delete '%s' if necessary"
                        % self.lock_file,
                        stacklevel=1 + stacklevel)
            if attempts > 3 * 60:
                raise RuntimeError("waited more than three minutes "
                        "on the lock file '%s'"
                        "--something is wrong" % self.lock_file)

        cleanup_m.register(self)

    def clean_up(self):
        os.close(self.fd)
        os.unlink(self.lock_file)

    def error_clean_up(self):
        pass


class ItemDirManager(CleanupBase):
    def __init__(self, cleanup_m, path, delete_on_error):
        from os.path import isdir

        self.existed = isdir(path)
        self.path = path
        self.delete_on_error = delete_on_error

        cleanup_m.register(self)

    def reset(self):
        try:
            shutil.rmtree(self.path)
        except OSError as e:
            if e.errno != errno.ENOENT:
                raise

    def mkdir(self):
        from os import mkdir
        try:
            mkdir(self.path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    def clean_up(self):
        pass

    def error_clean_up(self):
        if self.delete_on_error:
            self.reset()

# }}}


# {{{ key generation

class KeyBuilder(object):
    def rec(self, key_hash, key):
        digest = None

        try:
            digest = key._pytools_persistent_hash_digest  # noqa pylint:disable=protected-access
        except AttributeError:
            pass

        if digest is None:
            try:
                method = key.update_persistent_hash
            except AttributeError:
                pass
            else:
                inner_key_hash = new_hash()
                method(inner_key_hash, self)
                digest = inner_key_hash.digest()

        if digest is None:
            try:
                method = getattr(self, "update_for_"+type(key).__name__)
            except AttributeError:
                pass
            else:
                inner_key_hash = new_hash()
                method(inner_key_hash, key)
                digest = inner_key_hash.digest()

        if digest is None:
            raise TypeError("unsupported type for persistent hash keying: %s"
                    % type(key))

        if not isinstance(key, type):
            try:
                key._pytools_persistent_hash_digest = digest   # noqa pylint:disable=protected-access
            except AttributeError:
                pass
            except TypeError:
                pass

        key_hash.update(digest)

    def __call__(self, key):
        key_hash = new_hash()
        self.rec(key_hash, key)
        return key_hash.hexdigest()

    # {{{ updaters

    @staticmethod
    def update_for_int(key_hash, key):
        key_hash.update(str(key).encode("utf8"))

    update_for_long = update_for_int
    update_for_bool = update_for_int

    @staticmethod
    def update_for_float(key_hash, key):
        key_hash.update(repr(key).encode("utf8"))

    if sys.version_info >= (3,):
        @staticmethod
        def update_for_str(key_hash, key):
            key_hash.update(key.encode('utf8'))

        @staticmethod
        def update_for_bytes(key_hash, key):
            key_hash.update(key)
    else:
        @staticmethod
        def update_for_str(key_hash, key):
            key_hash.update(key)

        @staticmethod
        def update_for_unicode(key_hash, key):
            key_hash.update(key.encode('utf8'))

    def update_for_tuple(self, key_hash, key):
        for obj_i in key:
            self.rec(key_hash, obj_i)

    def update_for_frozenset(self, key_hash, key):
        for set_key in sorted(key):
            self.rec(key_hash, set_key)

    @staticmethod
    def update_for_NoneType(key_hash, key):  # noqa
        del key
        key_hash.update("<None>".encode('utf8'))

    @staticmethod
    def update_for_dtype(key_hash, key):
        key_hash.update(key.str.encode('utf8'))

    # }}}

# }}}


# {{{ lru cache

class _LinkedList(object):
    """The list operates on nodes of the form [value, leftptr, rightpr]. To create a
    node of this form you can use `LinkedList.new_node().`

    Supports inserting at the left and deleting from an arbitrary location.
    """
    def __init__(self):
        self.count = 0
        self.head = None
        self.end = None

    @staticmethod
    def new_node(element):
        return [element, None, None]

    def __len__(self):
        return self.count

    def appendleft_node(self, node):
        self.count += 1

        if self.head is None:
            self.head = self.end = node
            return

        self.head[1] = node
        node[2] = self.head

        self.head = node

    def pop_node(self):
        end = self.end
        self.remove_node(end)
        return end

    def remove_node(self, node):
        self.count -= 1

        if self.head is self.end:
            assert node is self.head
            self.head = self.end = None
            return

        left = node[1]
        right = node[2]

        if left is None:
            self.head = right
        else:
            left[2] = right

        if right is None:
            self.end = left
        else:
            right[1] = left

        node[1] = node[2] = None


class _LRUCache(abc.MutableMapping):
    """A mapping that keeps at most *maxsize* items with an LRU replacement policy.
    """
    def __init__(self, maxsize):
        self.lru_order = _LinkedList()
        self.maxsize = maxsize
        self.cache = {}

    def __delitem__(self, item):
        node = self.cache[item]
        self.lru_order.remove_node(node)
        del self.cache[item]

    def __getitem__(self, item):
        node = self.cache[item]
        self.lru_order.remove_node(node)
        self.lru_order.appendleft_node(node)
        # A linked list node contains a tuple of the form (item, value).
        return node[0][1]

    def __contains__(self, item):
        return item in self.cache

    def __iter__(self):
        return iter(self.cache)

    def __len__(self):
        return len(self.cache)

    def clear(self):
        self.cache.clear()
        self.lru_order = _LinkedList()

    def __setitem__(self, item, value):
        if self.maxsize < 1:
            return

        try:
            node = self.cache[item]
            self.lru_order.remove_node(node)
        except KeyError:
            if len(self.lru_order) >= self.maxsize:
                # Make room for new elements.
                end_node = self.lru_order.pop_node()
                del self.cache[end_node[0][0]]

            node = self.lru_order.new_node((item, value))
            self.cache[item] = node

        self.lru_order.appendleft_node(node)

        assert len(self.cache) == len(self.lru_order), \
                (len(self.cache), len(self.lru_order))
        assert len(self.lru_order) <= self.maxsize

# }}}


# {{{ top-level

class NoSuchEntryError(KeyError):
    pass


class ReadOnlyEntryError(KeyError):
    pass


class CollisionWarning(UserWarning):
    pass


class _PersistentDictBase(object):
    def __init__(self, identifier, key_builder=None, container_dir=None):
        self.identifier = identifier

        if key_builder is None:
            key_builder = KeyBuilder()

        self.key_builder = key_builder

        from os.path import join
        if container_dir is None:
            import appdirs
            container_dir = join(
                    appdirs.user_cache_dir("pytools", "pytools"),
                    "pdict-v2-%s-py%s" % (
                        identifier,
                        ".".join(str(i) for i in sys.version_info),))

        self.container_dir = container_dir

        self._make_container_dir()

    @staticmethod
    def _warn(msg, category=UserWarning, stacklevel=0):
        from warnings import warn
        warn(msg, category, stacklevel=1 + stacklevel)

    def store_if_not_present(self, key, value, _stacklevel=0):
        self.store(key, value, _skip_if_present=True, _stacklevel=1 + _stacklevel)

    def store(self, key, value, _skip_if_present=False, _stacklevel=0):
        raise NotImplementedError()

    def fetch(self, key, _stacklevel=0):
        raise NotImplementedError()

    @staticmethod
    def _read(path):
        from six.moves.cPickle import load
        with open(path, "rb") as inf:
            return load(inf)

    @staticmethod
    def _write(path, value):
        from six.moves.cPickle import dump, HIGHEST_PROTOCOL
        with open(path, "wb") as outf:
            dump(value, outf, protocol=HIGHEST_PROTOCOL)

    def _item_dir(self, hexdigest_key):
        from os.path import join
        return join(self.container_dir, hexdigest_key)

    def _key_file(self, hexdigest_key):
        from os.path import join
        return join(self._item_dir(hexdigest_key), "key")

    def _contents_file(self, hexdigest_key):
        from os.path import join
        return join(self._item_dir(hexdigest_key), "contents")

    def _lock_file(self, hexdigest_key):
        from os.path import join
        return join(self.container_dir, str(hexdigest_key) + ".lock")

    def _make_container_dir(self):
        _make_dir_recursively(self.container_dir)

    def _collision_check(self, key, stored_key, _stacklevel):
        if stored_key != key:
            # Key collision, oh well.
            self._warn("%s: key collision in cache at '%s' -- these are "
                    "sufficiently unlikely that they're often "
                    "indicative of a broken hash key implementation "
                    "(that is not considering some elements relevant "
                    "for equality comparison)"
                    % (self.identifier, self.container_dir),
                    CollisionWarning,
                    1 + _stacklevel)

            # This is here so we can step through equality comparison to
            # see what is actually non-equal.
            stored_key == key  # pylint:disable=pointless-statement
            raise NoSuchEntryError(key)

    def __getitem__(self, key):
        return self.fetch(key, _stacklevel=1)

    def __setitem__(self, key, value):
        self.store(key, value, _stacklevel=1)

    def clear(self):
        try:
            shutil.rmtree(self.container_dir)
        except OSError as e:
            if e.errno != errno.ENOENT:
                raise

        self._make_container_dir()


class WriteOncePersistentDict(_PersistentDictBase):
    """A concurrent disk-backed dictionary that disallows overwriting/deletion.

    Compared with :class:`PersistentDict`, this class has faster
    retrieval times.

    .. automethod:: __init__
    .. automethod:: __getitem__
    .. automethod:: __setitem__
    .. automethod:: clear
    .. automethod:: store
    .. automethod:: store_if_not_present
    .. automethod:: fetch
    """
    def __init__(self, identifier, key_builder=None, container_dir=None,
             in_mem_cache_size=256):
        """
        :arg identifier: a file-name-compatible string identifying this
            dictionary
        :arg key_builder: a subclass of :class:`KeyBuilder`
        :arg in_mem_cache_size: retain an in-memory cache of up to
            *in_mem_cache_size* items
        """
        _PersistentDictBase.__init__(self, identifier, key_builder, container_dir)
        self._cache = _LRUCache(in_mem_cache_size)

    def _spin_until_removed(self, lock_file, stacklevel):
        from os.path import exists

        attempts = 0
        while exists(lock_file):
            from time import sleep
            sleep(1)

            attempts += 1

            if attempts > 10:
                self._warn("waiting until unlocked--delete '%s' if necessary"
                        % lock_file, stacklevel=1 + stacklevel)

            if attempts > 3 * 60:
                raise RuntimeError("waited more than three minutes "
                        "on the lock file '%s'"
                        "--something is wrong" % lock_file)

    def store(self, key, value, _skip_if_present=False, _stacklevel=0):
        hexdigest_key = self.key_builder(key)

        cleanup_m = CleanupManager()
        try:
            try:
                LockManager(cleanup_m, self._lock_file(hexdigest_key),
                        1 + _stacklevel)
                item_dir_m = ItemDirManager(
                        cleanup_m, self._item_dir(hexdigest_key),
                        delete_on_error=False)

                if item_dir_m.existed:
                    if _skip_if_present:
                        return
                    raise ReadOnlyEntryError(key)

                item_dir_m.mkdir()

                key_path = self._key_file(hexdigest_key)
                value_path = self._contents_file(hexdigest_key)

                self._write(value_path, value)
                self._write(key_path, key)

                logger.debug("%s: disk cache store [key=%s]",
                        self.identifier, hexdigest_key)
            except Exception:
                cleanup_m.error_clean_up()
                raise
        finally:
            cleanup_m.clean_up()

    def fetch(self, key, _stacklevel=0):
        hexdigest_key = self.key_builder(key)

        # {{{ in memory cache

        try:
            stored_key, stored_value = self._cache[hexdigest_key]
        except KeyError:
            pass
        else:
            logger.debug("%s: in mem cache hit [key=%s]",
                    self.identifier, hexdigest_key)
            self._collision_check(key, stored_key, 1 + _stacklevel)
            return stored_value

        # }}}

        # {{{ check path exists and is unlocked

        item_dir = self._item_dir(hexdigest_key)

        from os.path import isdir
        if not isdir(item_dir):
            logger.debug("%s: disk cache miss [key=%s]",
                    self.identifier, hexdigest_key)
            raise NoSuchEntryError(key)

        lock_file = self._lock_file(hexdigest_key)
        self._spin_until_removed(lock_file, 1 + _stacklevel)

        # }}}

        key_file = self._key_file(hexdigest_key)
        contents_file = self._contents_file(hexdigest_key)

        # Note: Unlike PersistentDict, this doesn't autodelete invalid entires,
        # because that would lead to a race condition.

        # {{{ load key file and do equality check

        try:
            read_key = self._read(key_file)
        except Exception as e:
            self._warn("pytools.persistent_dict.WriteOncePersistentDict(%s) "
                    "encountered an invalid "
                    "key file for key %s. Remove the directory "
                    "'%s' if necessary. (caught: %s: %s)"
                    % (self.identifier, hexdigest_key, item_dir,
                        type(e).__name__, str(e)),
                    stacklevel=1 + _stacklevel)
            raise NoSuchEntryError(key)

        self._collision_check(key, read_key, 1 + _stacklevel)

        # }}}

        logger.debug("%s: disk cache hit [key=%s]",
                self.identifier, hexdigest_key)

        # {{{ load contents

        try:
            read_contents = self._read(contents_file)
        except Exception:
            self._warn("pytools.persistent_dict.WriteOncePersistentDict(%s) "
                    "encountered an invalid "
                    "key file for key %s. Remove the directory "
                    "'%s' if necessary."
                    % (self.identifier, hexdigest_key, item_dir),
                    stacklevel=1 + _stacklevel)
            raise NoSuchEntryError(key)

        # }}}

        self._cache[hexdigest_key] = (key, read_contents)
        return read_contents

    def clear(self):
        _PersistentDictBase.clear(self)
        self._cache.clear()


class PersistentDict(_PersistentDictBase):
    """A concurrent disk-backed dictionary.

    .. automethod:: __init__
    .. automethod:: __getitem__
    .. automethod:: __setitem__
    .. automethod:: __delitem__
    .. automethod:: clear
    .. automethod:: store
    .. automethod:: store_if_not_present
    .. automethod:: fetch
    .. automethod:: remove
    """
    def __init__(self, identifier, key_builder=None, container_dir=None):
        """
        :arg identifier: a file-name-compatible string identifying this
            dictionary
        :arg key_builder: a subclass of :class:`KeyBuilder`
        """
        _PersistentDictBase.__init__(self, identifier, key_builder, container_dir)

    def store(self, key, value, _skip_if_present=False, _stacklevel=0):
        hexdigest_key = self.key_builder(key)

        cleanup_m = CleanupManager()
        try:
            try:
                LockManager(cleanup_m, self._lock_file(hexdigest_key),
                        1 + _stacklevel)
                item_dir_m = ItemDirManager(
                        cleanup_m, self._item_dir(hexdigest_key),
                        delete_on_error=True)

                if item_dir_m.existed:
                    if _skip_if_present:
                        return
                    item_dir_m.reset()

                item_dir_m.mkdir()

                key_path = self._key_file(hexdigest_key)
                value_path = self._contents_file(hexdigest_key)

                self._write(value_path, value)
                self._write(key_path, key)

                logger.debug("%s: cache store [key=%s]",
                        self.identifier, hexdigest_key)
            except Exception:
                cleanup_m.error_clean_up()
                raise
        finally:
            cleanup_m.clean_up()

    def fetch(self, key, _stacklevel=0):
        hexdigest_key = self.key_builder(key)
        item_dir = self._item_dir(hexdigest_key)

        from os.path import isdir
        if not isdir(item_dir):
            logger.debug("%s: cache miss [key=%s]",
                    self.identifier, hexdigest_key)
            raise NoSuchEntryError(key)

        cleanup_m = CleanupManager()
        try:
            try:
                LockManager(cleanup_m, self._lock_file(hexdigest_key),
                        1 + _stacklevel)
                item_dir_m = ItemDirManager(
                        cleanup_m, item_dir, delete_on_error=False)

                key_path = self._key_file(hexdigest_key)
                value_path = self._contents_file(hexdigest_key)

                # {{{ load key

                try:
                    read_key = self._read(key_path)
                except Exception:
                    item_dir_m.reset()
                    self._warn("pytools.persistent_dict.PersistentDict(%s) "
                            "encountered an invalid "
                            "key file for key %s. Entry deleted."
                            % (self.identifier, hexdigest_key),
                            stacklevel=1 + _stacklevel)
                    raise NoSuchEntryError(key)

                self._collision_check(key, read_key, 1 + _stacklevel)

                # }}}

                logger.debug("%s: cache hit [key=%s]",
                        self.identifier, hexdigest_key)

                # {{{ load value

                try:
                    read_contents = self._read(value_path)
                except Exception:
                    item_dir_m.reset()
                    self._warn("pytools.persistent_dict.PersistentDict(%s) "
                            "encountered an invalid "
                            "key file for key %s. Entry deleted."
                            % (self.identifier, hexdigest_key),
                            stacklevel=1 + _stacklevel)
                    raise NoSuchEntryError(key)

                return read_contents

                # }}}

            except Exception:
                cleanup_m.error_clean_up()
                raise
        finally:
            cleanup_m.clean_up()

    def remove(self, key, _stacklevel=0):
        hexdigest_key = self.key_builder(key)

        item_dir = self._item_dir(hexdigest_key)
        from os.path import isdir
        if not isdir(item_dir):
            raise NoSuchEntryError(key)

        cleanup_m = CleanupManager()
        try:
            try:
                LockManager(cleanup_m, self._lock_file(hexdigest_key),
                        1 + _stacklevel)
                item_dir_m = ItemDirManager(
                        cleanup_m, item_dir, delete_on_error=False)
                key_file = self._key_file(hexdigest_key)

                # {{{ load key

                try:
                    read_key = self._read(key_file)
                except Exception:
                    item_dir_m.reset()
                    self._warn("pytools.persistent_dict.PersistentDict(%s) "
                            "encountered an invalid "
                            "key file for key %s. Entry deleted."
                            % (self.identifier, hexdigest_key),
                            stacklevel=1 + _stacklevel)
                    raise NoSuchEntryError(key)

                self._collision_check(key, read_key, 1 + _stacklevel)

                # }}}

                item_dir_m.reset()

            except Exception:
                cleanup_m.error_clean_up()
                raise
        finally:
            cleanup_m.clean_up()

    def __delitem__(self, key):
        self.remove(key, _stacklevel=1)

# }}}

# vim: foldmethod=marker
