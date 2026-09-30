"""Exception and warning classes for malformed ARL file content."""


class ARLFormatError(ValueError):
    """
    Raised when file content is not valid ARL.

    Examples are an unparseable record header or index record, a file whose
    size is not a whole number of records, or a time step that is repeated
    with different content. Subclasses ``ValueError`` so existing
    ``except ValueError`` handlers still catch it.

    Bad arguments (programmer errors) still raise plain ``ValueError`` or
    ``TypeError``; this class is only for problems with the bytes on disk.
    """


class ARLFormatWarning(UserWarning):
    """
    Warning for recoverable irregularities in ARL file content.

    For example, some NOAA archive files repeat a whole time step with
    byte-identical content; the repeated copy is ignored with this warning.
    """
