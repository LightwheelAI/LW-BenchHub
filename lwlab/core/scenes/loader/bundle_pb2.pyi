from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor


class S3BucketType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    S3_BUCKET_TYPE_UNSPECIFIED: _ClassVar[S3BucketType]
    S3_BUCKET_TYPE_US3_DEFAULT: _ClassVar[S3BucketType]


S3_BUCKET_TYPE_UNSPECIFIED: S3BucketType
S3_BUCKET_TYPE_US3_DEFAULT: S3BucketType


class GetBundleRequest(_message.Message):
    __slots__ = ("layout_id", "style_id", "scene")
    LAYOUT_ID_FIELD_NUMBER: _ClassVar[int]
    STYLE_ID_FIELD_NUMBER: _ClassVar[int]
    SCENE_FIELD_NUMBER: _ClassVar[int]
    layout_id: int
    style_id: int
    scene: str
    def __init__(self, layout_id: _Optional[int] = ..., style_id: _Optional[int] = ..., scene: _Optional[str] = ...) -> None: ...


class GetBundleReply(_message.Message):
    __slots__ = ("data",)
    DATA_FIELD_NUMBER: _ClassVar[int]
    data: bytes
    def __init__(self, data: _Optional[bytes] = ...) -> None: ...


class CreateBundleRequest(_message.Message):
    __slots__ = ("uuid", "layout_id", "style_id", "file_name", "data_path", "scene", "s3_bucket_type", "image", "svn", "worker_version")
    UUID_FIELD_NUMBER: _ClassVar[int]
    LAYOUT_ID_FIELD_NUMBER: _ClassVar[int]
    STYLE_ID_FIELD_NUMBER: _ClassVar[int]
    FILE_NAME_FIELD_NUMBER: _ClassVar[int]
    DATA_PATH_FIELD_NUMBER: _ClassVar[int]
    SCENE_FIELD_NUMBER: _ClassVar[int]
    S3_BUCKET_TYPE_FIELD_NUMBER: _ClassVar[int]
    IMAGE_FIELD_NUMBER: _ClassVar[int]
    SVN_FIELD_NUMBER: _ClassVar[int]
    WORKER_VERSION_FIELD_NUMBER: _ClassVar[int]
    uuid: str
    layout_id: int
    style_id: int
    file_name: str
    data_path: str
    scene: str
    s3_bucket_type: S3BucketType
    image: str
    svn: int
    worker_version: str
    def __init__(self, uuid: _Optional[str] = ..., layout_id: _Optional[int] = ..., style_id: _Optional[int] = ..., file_name: _Optional[str] = ..., data_path: _Optional[str] = ..., scene: _Optional[str] = ..., s3_bucket_type: _Optional[_Union[S3BucketType, str]] = ..., image: _Optional[str] = ..., svn: _Optional[int] = ..., worker_version: _Optional[str] = ...) -> None: ...


class CreateBundleReply(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...


class GetVersionRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...


class GetVersionReply(_message.Message):
    __slots__ = ("version",)
    VERSION_FIELD_NUMBER: _ClassVar[int]
    version: str
    def __init__(self, version: _Optional[str] = ...) -> None: ...


class UpdateVersionRequest(_message.Message):
    __slots__ = ("image", "svn", "worker_version")
    IMAGE_FIELD_NUMBER: _ClassVar[int]
    SVN_FIELD_NUMBER: _ClassVar[int]
    WORKER_VERSION_FIELD_NUMBER: _ClassVar[int]
    image: str
    svn: int
    worker_version: str
    def __init__(self, image: _Optional[str] = ..., svn: _Optional[int] = ..., worker_version: _Optional[str] = ...) -> None: ...


class UpdateVersionReply(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...
