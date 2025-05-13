import json
import os
import queue
import shutil
import subprocess
import threading
import warnings
import uuid
from typing import Type, TypeVar, Generic, List, Optional, Literal, Dict, Set, Any, Union
from datetime import datetime, date
import hashlib
import secrets
import base64

import sqlalchemy as sa
from sqlalchemy import create_engine, Column, Integer, String, Text, Boolean, ForeignKey, DateTime, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.sql.expression import func

from pydantic import BaseModel, validator, root_validator
from pydantic.json import pydantic_encoder

# Define a generic type variable for the Pydantic model
T = TypeVar('T', bound=BaseModel)

# SQLAlchemy models
Base = declarative_base()

# Junction table for file tags
file_tags = Table(
    'file_tags',
    Base.metadata,
    Column('file_id', String(36), ForeignKey('files.id')),
    Column('tag_id', Integer, ForeignKey('tags.id'))
)


class User(Base):
    __tablename__ = 'users'

    id = Column(String(36), primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    # Store a hashed password
    password_hash = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_admin = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)

    # Relationships
    files = relationship("File", back_populates="owner")

    def __repr__(self):
        return f"<User {self.username}>"


class Tag(Base):
    __tablename__ = 'tags'

    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True, nullable=False)

    files = relationship("File", secondary=file_tags, back_populates="tags")

    def __repr__(self):
        return f"<Tag {self.name}>"


class File(Base):
    __tablename__ = 'files'

    id = Column(String(36), primary_key=True)
    file_name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    file_type = Column(String(20), nullable=False)
    created_date = Column(DateTime, default=datetime.utcnow)
    modified_date = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    date = Column(DateTime, nullable=False)  # The user-specified date
    file_path = Column(String(255), nullable=False)  # Physical path on disk

    # File type-specific fields
    # For ModelFile: 'uploaded' or 'trained'
    source = Column(String(20), nullable=True)
    # For ModelFile and SearchFile: 'pending', 'running', 'completed', 'failed'
    status = Column(String(20), nullable=True)
    # For ModelFile: reference to a config file
    config_id = Column(String(36), ForeignKey('files.id'), nullable=True)
    enzyme = Column(String(50), nullable=True)  # For SpectraFile
    instrument = Column(String(50), nullable=True)  # For SpectraFile
    annotated = Column(Boolean, nullable=True)  # For SpectraFile
    # For SearchFile: reference to a model file
    model_id = Column(String(36), ForeignKey('files.id'), nullable=True)
    # For SearchFile: reference to a spectra file
    spectra_id = Column(String(36), ForeignKey('files.id'), nullable=True)

    # Owner relationship
    owner_id = Column(String(36), ForeignKey('users.id'), nullable=False)
    owner = relationship("User", back_populates="files")

    # Tags relationship
    tags = relationship("Tag", secondary=file_tags, back_populates="files")

    # Self-references
    config = relationship("File", foreign_keys=[config_id], remote_side=[id], uselist=False,
                          primaryjoin=(config_id == id), post_update=True)
    model = relationship("File", foreign_keys=[model_id], remote_side=[id], uselist=False,
                         primaryjoin=(model_id == id), post_update=True)
    spectra = relationship("File", foreign_keys=[spectra_id], remote_side=[id], uselist=False,
                           primaryjoin=(spectra_id == id), post_update=True)

    # 'model', 'spectra', 'search', 'config'
    file_type_enum = Column(String(20), nullable=False)

    __mapper_args__ = {
        'polymorphic_on': file_type_enum,
        'polymorphic_identity': 'file'
    }

    def __repr__(self):
        return f"<File {self.file_name} ({self.id})>"

    def to_dict(self):
        """Convert file object to a dictionary"""
        result = {
            'file_id': self.id,
            'file_name': self.file_name,
            'description': self.description,
            'file_type': self.file_type,
            'date': self.date,
            'tags': [tag.name for tag in self.tags],
            'owner_id': self.owner_id,
        }

        # Add type-specific fields based on file_type_enum
        if self.file_type_enum == 'model':
            result.update({
                'source': self.source,
                'status': self.status,
                'config': self.config_id,
            })
        elif self.file_type_enum == 'spectra':
            result.update({
                'enzyme': self.enzyme,
                'instrument': self.instrument,
                'annotated': self.annotated,
            })
        elif self.file_type_enum == 'search':
            result.update({
                'model': self.model_id,
                'spectra': self.spectra_id,
                'status': self.status,
            })

        return result


# Add concrete polymorphic subclasses for each file type
class ModelFile(File):
    __mapper_args__ = {
        'polymorphic_identity': 'model'
    }


class SpectraFile(File):
    __mapper_args__ = {
        'polymorphic_identity': 'spectra'
    }


class SearchFile(File):
    __mapper_args__ = {
        'polymorphic_identity': 'search'
    }


class ConfigFile(File):
    __mapper_args__ = {
        'polymorphic_identity': 'config'
    }


# Pydantic models for validation (same as in simple_db)
class UserModel(BaseModel):
    id: str
    username: str
    email: str
    is_admin: bool = False
    is_active: bool = True

    @validator('id')
    def validate_id(cls, v):
        try:
            uuid_obj = uuid.UUID(v)
            return str(uuid_obj)
        except ValueError:
            raise ValueError('Invalid UUID format')

    class Config:
        orm_mode = True


class GeneralFileMetadata(BaseModel):
    file_id: str
    file_name: str
    description: str
    file_type: str
    date: date
    tags: List[str]
    owner_id: Optional[str] = None

    class Config:
        orm_mode = True


class ModelFileMetadata(GeneralFileMetadata):
    source: Literal['uploaded', 'trained']
    status: Literal['pending', 'running', 'completed', 'failed']
    config: Optional[str]


class SpectraFileMetadata(GeneralFileMetadata):
    enzyme: str
    instrument: str
    annotated: bool


class SearchMetadata(GeneralFileMetadata):
    model: Optional[str]
    spectra: Optional[str]
    status: Literal['pending', 'running', 'completed', 'failed']


class ConfigFileMetadata(GeneralFileMetadata):
    pass


class DBConnectionManager:
    """Singleton class to manage database connections"""
    _instance = None

    def __new__(cls, db_path=None):
        if cls._instance is None:
            cls._instance = super(DBConnectionManager, cls).__new__(cls)
            if db_path:
                # SQLite connection string
                connection_string = f"sqlite:///{db_path}"
                cls._instance.engine = create_engine(
                    connection_string, echo=False)
                Base.metadata.create_all(cls._instance.engine)
                cls._instance.Session = sessionmaker(bind=cls._instance.engine)
        return cls._instance

    def get_session(self) -> Session:
        """Get a new session"""
        return self.Session()

    def create_default_admin(self):
        """Create a default admin user if no users exist"""
        with self.get_session() as session:
            if session.query(User).count() == 0:
                # Simple password hash - in production, use a proper hashing library
                admin_id = str(uuid.uuid4())
                admin = User(
                    id=admin_id,
                    username="admin",
                    email="admin@example.com",
                    password_hash="admin123_hash",  # In production, use a proper hash
                    is_admin=True
                )
                session.add(admin)
                session.commit()
                return admin_id
            else:
                # Return the ID of an existing admin
                admin = session.query(User).filter_by(is_admin=True).first()
                if admin:
                    return admin.id
                else:
                    # No admin exists, but users do - create a new admin
                    admin_id = str(uuid.uuid4())
                    admin = User(
                        id=admin_id,
                        username="admin",
                        email="admin@example.com",
                        password_hash="admin123_hash",  # In production, use a proper hash
                        is_admin=True
                    )
                    session.add(admin)
                    session.commit()
                    return admin_id


class _SQLiteManager(Generic[T]):
    """Manages database operations for a specific model type"""

    def __init__(self,
                 model: Type[T],
                 db_connection: DBConnectionManager,
                 file_type: str,
                 lock: threading.Lock,
                 verbose: bool,
                 ):
        self.model = model
        self.db_connection = db_connection
        self.file_type = file_type  # 'model', 'spectra', 'search', 'config'
        self.verbose = verbose
        self.lock = lock

    def _log(self, message: str):
        if self.verbose:
            print(f'SQLiteManager[{self.model.__name__}]: {message}')

    def _get_or_create_tags(self, session: Session, tag_names: List[str]) -> List[Tag]:
        """Get existing tags or create new ones"""
        tags = []
        for name in tag_names:
            tag = session.query(Tag).filter_by(name=name).first()
            if not tag:
                tag = Tag(name=name)
                session.add(tag)
            tags.append(tag)
        return tags

    def add_metadata(self, metadata: T, owner_id: str = None):
        """Add file metadata to the database"""
        with self.lock:
            session = self.db_connection.get_session()
            try:
                # Convert Pydantic model to dict
                data = metadata.dict()

                # Create tags if they don't exist
                tags = self._get_or_create_tags(session, data.pop('tags', []))

                # Handle special references
                config_id = data.pop('config', None)
                model_id = data.pop('model', None)
                spectra_id = data.pop('spectra', None)

                # Create File object
                file_obj = File(
                    id=data['file_id'],
                    file_name=data['file_name'],
                    description=data.get('description', ''),
                    file_type=data['file_type'],
                    date=data['date'],
                    file_type_enum=self.file_type,
                    tags=tags,
                    owner_id=owner_id,
                    # Type-specific fields
                    source=data.get('source'),
                    status=data.get('status'),
                    config_id=config_id,
                    enzyme=data.get('enzyme'),
                    instrument=data.get('instrument'),
                    annotated=data.get('annotated'),
                    model_id=model_id,
                    spectra_id=spectra_id,
                    # Will be updated later
                    file_path=''
                )

                session.add(file_obj)
                session.commit()
                self._log(f"Added metadata: {data}")
                return file_obj.id
            except Exception as e:
                session.rollback()
                self._log(f"Error adding metadata: {e}")
                raise
            finally:
                session.close()

    def update_metadata(self, metadata: T):
        """Update file metadata in the database"""
        with self.lock:
            session = self.db_connection.get_session()
            try:
                # Convert Pydantic model to dict
                data = metadata.dict()
                file_id = data.pop('file_id')

                # Get the file object
                file_obj = session.query(File).filter_by(id=file_id).first()
                if not file_obj:
                    raise FileNotFoundError(
                        f"File metadata for '{file_id}' not found.")

                # Update tags
                if 'tags' in data:
                    tags = self._get_or_create_tags(
                        session, data.pop('tags', []))
                    file_obj.tags = tags

                # Handle special references
                if 'config' in data:
                    file_obj.config_id = data.pop('config', None)
                if 'model' in data:
                    file_obj.model_id = data.pop('model', None)
                if 'spectra' in data:
                    file_obj.spectra_id = data.pop('spectra', None)

                # Update general fields
                for key, value in data.items():
                    if hasattr(file_obj, key):
                        setattr(file_obj, key, value)

                file_obj.modified_date = datetime.utcnow()
                session.commit()
                self._log(f"Updated metadata for file_id: {file_id}")
            except Exception as e:
                session.rollback()
                self._log(f"Error updating metadata: {e}")
                raise
            finally:
                session.close()

    def delete_metadata(self, file_id: str):
        """Delete file metadata from the database"""
        with self.lock:
            session = self.db_connection.get_session()
            try:
                file_obj = session.query(File).filter_by(id=file_id).first()
                if file_obj:
                    session.delete(file_obj)
                    session.commit()
                    self._log(f"Deleted metadata for file_id: {file_id}")
                else:
                    self._log(f"No metadata found for file_id: {file_id}")
            except Exception as e:
                session.rollback()
                self._log(f"Error deleting metadata: {e}")
                raise
            finally:
                session.close()

    def get_metadata(self, file_id: str, model: Type[T]) -> Optional[T]:
        """Get file metadata from the database"""
        session = self.db_connection.get_session()
        try:
            file_obj = session.query(File).filter_by(id=file_id).first()
            if file_obj:
                # Convert SQLAlchemy object to dict suitable for Pydantic model
                data = file_obj.to_dict()
                self._log(f"Retrieved metadata for file_id: {file_id}")
                return model(**data)
            else:
                self._log(f"No metadata found for file_id: {file_id}")
                return None
        except Exception as e:
            self._log(f"Error retrieving metadata: {e}")
            raise
        finally:
            session.close()

    def get_all_metadata(self, owner_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all file metadata from the database, optionally filtered by owner"""
        session = self.db_connection.get_session()
        try:
            query = session.query(File).filter(
                File.file_type_enum == self.file_type)
            if owner_id:
                query = query.filter(File.owner_id == owner_id)

            file_objs = query.all()
            result = [file_obj.to_dict() for file_obj in file_objs]
            self._log(f"Retrieved {len(result)} metadata records")
            return result
        except Exception as e:
            self._log(f"Error retrieving all metadata: {e}")
            raise
        finally:
            session.close()

    def update_file_path(self, file_id: str, file_path: str):
        """Update the file path in the database"""
        with self.lock:
            session = self.db_connection.get_session()
            try:
                file_obj = session.query(File).filter_by(id=file_id).first()
                if file_obj:
                    file_obj.file_path = file_path
                    session.commit()
                    self._log(f"Updated file path for file_id: {file_id}")
                else:
                    raise FileNotFoundError(
                        f"File metadata for '{file_id}' not found.")
            except Exception as e:
                session.rollback()
                self._log(f"Error updating file path: {e}")
                raise
            finally:
                session.close()

    def search_by_tags(self, tags: List[str], require_all: bool = False) -> List[Dict[str, Any]]:
        """Search files by tags"""
        session = self.db_connection.get_session()
        try:
            query = session.query(File).filter(
                File.file_type_enum == self.file_type)

            if tags:
                if require_all:
                    # Files must have all specified tags
                    for tag in tags:
                        query = query.filter(File.tags.any(Tag.name == tag))
                else:
                    # Files must have any of the specified tags
                    query = query.filter(File.tags.any(Tag.name.in_(tags)))

            file_objs = query.all()
            result = [file_obj.to_dict() for file_obj in file_objs]
            self._log(f"Found {len(result)} files with tags: {tags}")
            return result
        except Exception as e:
            self._log(f"Error searching by tags: {e}")
            raise
        finally:
            session.close()

    def get_all_tables(self) -> List[str]:
        """Get all tables in the database"""
        return Base.metadata.tables.keys()

    def get_file_path(self, file_id: str) -> str:
        """Get the file path from the database"""
        session = self.db_connection.get_session()
        try:
            file_obj = session.query(File).filter_by(id=file_id).first()
            if file_obj:
                return file_obj.file_path
            else:
                raise FileNotFoundError(
                    f"File metadata for '{file_id}' not found.")
        except Exception as e:
            self._log(f"Error retrieving file path: {e}")
            raise
        finally:
            session.close()


class FileManager(Generic[T]):
    """Manages files and their metadata"""

    def __init__(self,
                 model: Type[T],
                 storage_dir: str,
                 db_connection: DBConnectionManager,
                 file_type: str,
                 lock: threading.Lock,
                 verbose: bool = False,
                 ):
        self.storage_dir = storage_dir
        self.db_manager = _SQLiteManager[T](
            model, db_connection, file_type, lock, verbose)
        self.model = model
        self.verbose = verbose
        os.makedirs(self.storage_dir, exist_ok=True)
        self.lock = threading.Lock()

    def _log(self, message: str):
        if self.verbose:
            print(f'FileManager[{self.model.__name__}]: {message}')

    def add_file(self, file_path: Optional[str], metadata: T, copy: bool = True, owner_id: str = None) -> str:
        """Add a file to storage and database"""
        file_id = metadata.file_id
        file_ext = metadata.file_type

        # Create user directory if needed
        user_storage_dir = os.path.join(
            self.storage_dir, owner_id if owner_id else "shared")
        os.makedirs(user_storage_dir, exist_ok=True)

        dest_path = os.path.join(user_storage_dir, f"{file_id}.{file_ext}")

        if os.path.exists(dest_path):
            raise FileExistsError(
                f"File '{file_id}' already exists in storage.")

        with self.lock:
            # First add metadata to get the file ID
            self.db_manager.add_metadata(metadata, owner_id)

            # Copy or move the file to storage
            if file_path is not None:
                if copy:
                    shutil.copy(file_path, dest_path)
                else:
                    shutil.move(file_path, dest_path)

            # Update the file path in the database
            self.db_manager.update_file_path(file_id, dest_path)

        self._log(
            f"Added file '{file_path}' as '{file_id}' with metadata {metadata}")
        return file_id

    def update_file_metadata(self, metadata: T) -> None:
        """Update file metadata"""
        with self.lock:
            if not self.db_manager.get_metadata(metadata.file_id, self.model):
                raise FileNotFoundError(
                    f"File metadata for '{metadata.file_id}' not found.")
            self.db_manager.update_metadata(metadata)
        self._log(f"Updated metadata for file '{metadata.file_id}'")

    def delete_file(self, file_id: str) -> None:
        """Delete a file and its metadata"""
        with self.lock:
            try:
                # Get file path before deleting metadata
                file_path = self.db_manager.get_file_path(file_id)

                # Delete metadata
                self.db_manager.delete_metadata(file_id)

                # Delete file if it exists
                if os.path.exists(file_path):
                    os.remove(file_path)
                else:
                    warnings.warn(
                        f"File '{file_id}' not found in storage at {file_path}.")

                self._log(f"Deleted file '{file_id}' and its metadata")
            except Exception as e:
                self._log(f"Error deleting file: {e}")
                raise

    def get_file_metadata(self, file_id: str) -> T:
        """Get file metadata"""
        metadata = self.db_manager.get_metadata(file_id, self.model)
        if not metadata:
            raise FileNotFoundError(
                f"File metadata for '{file_id}' not found.")
        self._log(f"Retrieved metadata for file '{file_id}'")
        return metadata

    def get_all_metadata(self, owner_id: Optional[str] = None) -> List[T]:
        """Get all file metadata, optionally filtered by owner"""
        metadata_dicts = self.db_manager.get_all_metadata(owner_id)
        metadata_list = [self.model(**data) for data in metadata_dicts]
        self._log(f"Retrieved {len(metadata_list)} metadata records")
        return metadata_list

    def get_all_file_ids(self, owner_id: Optional[str] = None) -> List[str]:
        """Get all file IDs, optionally filtered by owner"""
        metadata_dicts = self.db_manager.get_all_metadata(owner_id)
        return [data['file_id'] for data in metadata_dicts]

    def retrieve_file(self, file_id: str, dest_path: str) -> str:
        """Copy a file to a destination path"""
        with self.lock:
            src_path = self.db_manager.get_file_path(file_id)
            if not os.path.exists(src_path):
                raise FileNotFoundError(
                    f"File '{file_id}' not found in storage at {src_path}.")
            shutil.copy(src_path, dest_path)
        self._log(f"Retrieved file '{file_id}' to '{dest_path}'")
        return dest_path

    def retrieve_file_path(self, file_id: str) -> str:
        """Get the file path"""
        return self.db_manager.get_file_path(file_id)

    def search_by_tags(self, tags: List[str], require_all: bool = False) -> List[T]:
        """Search files by tags"""
        metadata_dicts = self.db_manager.search_by_tags(tags, require_all)
        metadata_list = [self.model(**data) for data in metadata_dicts]
        return metadata_list

    def reset_db(self):
        """Reset the database and delete all files for this file type"""
        try:
            with self.lock:
                # Get all file paths before deleting metadata
                file_paths = [self.db_manager.get_file_path(file_id)
                              for file_id in self.get_all_file_ids()]

                # Delete all files for this file type from the database
                for file_id in self.get_all_file_ids():
                    self.db_manager.delete_metadata(file_id)

                # Delete all files from storage
                for file_path in file_paths:
                    if os.path.exists(file_path):
                        os.remove(file_path)

            self._log("Reset the database and deleted all files")
        except Exception as e:
            self._log(f"Error resetting database: {e}")
            raise


class UserManager:
    """Manages user operations with secure password handling"""

    def __init__(self, db_connection: DBConnectionManager, verbose: bool = False):
        self.db_connection = db_connection
        self.verbose = verbose
        self.lock = threading.Lock()

    def _log(self, message: str):
        if self.verbose:
            print(f'UserManager: {message}')

    def _hash_password(self, password: str) -> str:
        """
        Securely hash a password using PBKDF2 with a random salt.

        Args:
            password: The plaintext password to hash

        Returns:
            A string in the format salt:hash that can be stored in the database
        """
        # Generate a random salt
        salt = secrets.token_hex(16)
        # Use PBKDF2 with SHA-256, 100,000 iterations (adjust based on security needs)
        key = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        )
        # Return salt and key as a single string, separated by colon
        return f"{salt}:{base64.b64encode(key).decode('utf-8')}"

    def _verify_password(self, stored_password: str, provided_password: str) -> bool:
        """
        Verify that a provided password matches the stored hash.

        Args:
            stored_password: The password hash retrieved from the database (salt:hash)
            provided_password: The plaintext password to verify

        Returns:
            True if the passwords match, False otherwise
        """
        try:
            # Split the stored password into salt and hash
            salt, stored_hash = stored_password.split(':', 1)
            # Hash the provided password with the same salt
            key = hashlib.pbkdf2_hmac(
                'sha256',
                provided_password.encode('utf-8'),
                salt.encode('utf-8'),
                100000
            )
            # Compare the computed hash with the stored hash
            provided_hash = base64.b64encode(key).decode('utf-8')
            return secrets.compare_digest(provided_hash, stored_hash)
        except Exception:
            # If anything goes wrong (e.g., invalid stored_password format), authentication fails
            return False

    def create_user(self, username: str, email: str, password: str, is_admin: bool = False) -> str:
        """Create a new user with a securely hashed password"""
        with self.lock:
            session = self.db_connection.get_session()
            try:
                # Check if username or email already exists
                if session.query(User).filter_by(username=username).first():
                    raise ValueError(f"Username '{username}' already exists")
                if session.query(User).filter_by(email=email).first():
                    raise ValueError(f"Email '{email}' already exists")

                # Create user with a secure password hash
                user_id = str(uuid.uuid4())
                user = User(
                    id=user_id,
                    username=username,
                    email=email,
                    password_hash=self._hash_password(password),
                    is_admin=is_admin
                )
                session.add(user)
                session.commit()
                self._log(f"Created user '{username}' with ID {user_id}")
                return user_id
            except Exception as e:
                session.rollback()
                self._log(f"Error creating user: {e}")
                raise
            finally:
                session.close()

    def authenticate_user(self, username: str, password: str) -> Optional[str]:
        """Authenticate a user using secure password verification and return user ID if successful"""
        session = self.db_connection.get_session()
        try:
            # Get the user by username
            user = session.query(User).filter_by(
                username=username,
                is_active=True
            ).first()

            if user and self._verify_password(user.password_hash, password):
                self._log(f"Authenticated user '{username}'")
                return user.id
            else:
                self._log(f"Authentication failed for user '{username}'")
                return None
        except Exception as e:
            self._log(f"Error authenticating user: {e}")
            raise
        finally:
            session.close()

    def change_password(self, user_id: str, current_password: str, new_password: str) -> bool:
        """Change user password with secure verification and hashing"""
        with self.lock:
            session = self.db_connection.get_session()
            try:
                user = session.query(User).filter_by(id=user_id).first()

                if not user:
                    return False

                # Verify current password
                if not self._verify_password(user.password_hash, current_password):
                    return False

                # Update with new securely hashed password
                user.password_hash = self._hash_password(new_password)
                session.commit()
                self._log(f"Changed password for user '{user.username}'")
                return True
            except Exception as e:
                session.rollback()
                self._log(f"Error changing password: {e}")
                raise
            finally:
                session.close()

    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID"""
        session = self.db_connection.get_session()
        try:
            user = session.query(User).filter_by(id=user_id).first()
            if user:
                return {
                    'id': user.id,
                    'username': user.username,
                    'email': user.email,
                    'is_admin': user.is_admin,
                    'is_active': user.is_active,
                    'created_at': user.created_at
                }
            else:
                return None
        finally:
            session.close()

    def update_user(self, user_id: str, **kwargs) -> bool:
        """Update user information"""
        with self.lock:
            session = self.db_connection.get_session()
            try:
                user = session.query(User).filter_by(id=user_id).first()
                if not user:
                    return False

                # Update only allowed fields
                allowed_fields = ['email', 'is_admin', 'is_active']
                for field, value in kwargs.items():
                    if field in allowed_fields:
                        setattr(user, field, value)

                session.commit()
                self._log(f"Updated user '{user.username}'")
                return True
            except Exception as e:
                session.rollback()
                self._log(f"Error updating user: {e}")
                raise
            finally:
                session.close()

    def list_users(self) -> List[Dict[str, Any]]:
        """List all users"""
        session = self.db_connection.get_session()
        try:
            users = session.query(User).all()
            return [
                {
                    'id': user.id,
                    'username': user.username,
                    'email': user.email,
                    'is_admin': user.is_admin,
                    'is_active': user.is_active,
                    'created_at': user.created_at
                }
                for user in users
            ]
        finally:
            session.close()


class CasanovoAdvancedDB:
    """
    Advanced database for Casanovo using SQLite and multi-user support.
    This class maintains API compatibility with the original CasanovoDB.
    """

    def __init__(self,
                 data_folder: str,
                 models_storage_folder: str = 'models',
                 models_table_name: str = 'models',
                 spectra_files_storage_folder: str = 'files',
                 config_storage_folder: str = 'config',
                 spectra_files_table_name: str = 'files',
                 searches_storage_folder: str = 'searches',
                 searches_table_name: str = 'searches',
                 config_table_name: str = 'config',
                 verbose: bool = False):

        # Create the data folder if it doesn't exist
        os.makedirs(data_folder, exist_ok=True)

        # Initialize the database connection
        db_path = os.path.join(data_folder, 'casanovo.db')
        self.db_connection = DBConnectionManager(db_path)

        # Create default admin user and get its ID
        self.default_user_id = self.db_connection.create_default_admin()

        # Create user manager
        self.user_manager = UserManager(self.db_connection, verbose)

        # Create storage folders
        models_storage_path = os.path.join(data_folder, models_storage_folder)
        spectra_files_storage_path = os.path.join(
            data_folder, spectra_files_storage_folder)
        searches_storage_path = os.path.join(
            data_folder, searches_storage_folder)
        config_storage_path = os.path.join(data_folder, config_storage_folder)

        # Create shared thread lock
        meta_data_lock = threading.Lock()

        # Initialize file managers
        self.models_manager = FileManager[ModelFileMetadata](
            ModelFileMetadata,
            models_storage_path,
            self.db_connection,
            'model',
            meta_data_lock,
            verbose
        )

        self.spectra_files_manager = FileManager[SpectraFileMetadata](
            SpectraFileMetadata,
            spectra_files_storage_path,
            self.db_connection,
            'spectra',
            meta_data_lock,
            verbose
        )

        self.searches_manager = FileManager[SearchMetadata](
            SearchMetadata,
            searches_storage_path,
            self.db_connection,
            'search',
            meta_data_lock,
            verbose
        )

        self.config_manager = FileManager[ConfigFileMetadata](
            ConfigFileMetadata,
            config_storage_path,
            self.db_connection,
            'config',
            meta_data_lock,
            verbose
        )

        # Queue setup for processing tasks
        self.verbose = verbose
        self.stop_event = threading.Event()
        self.queue = queue.Queue()
        self.current_task = None
        self.queue_thread = threading.Thread(target=self._process_queue)
        self.queue_thread.daemon = True
        self.queue_thread.start()
        self.update_unfinished_searches()

    def _process_queue(self):
        """Process tasks from the queue (same as in original CasanovoDB)"""
        while not self.stop_event.is_set():
            self.current_task = None
            try:
                task = self.queue.get()
                if task is None:
                    break

                if task['target'] == 'train':
                    self.current_task = task
                    self._run_train(
                        spectra_paths=task['spectra_paths'],
                        config_path=task['config_path'],
                        metadata=task['metadata']
                    )
                elif task['target'] == 'search':
                    self.current_task = task
                    self._run_search(
                        model_path=task['model_path'],
                        spectra_path=task['spectra_path'],
                        config_path=task['config_path'],
                        metadata=task['metadata']
                    )
                else:
                    raise ValueError(f"Invalid task target: {task['target']}")

                self.queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing task: {e}")
                # Handle task failure
                if self.current_task:
                    metadata = self.current_task['metadata']
                    metadata.status = 'failed'
                    self.searches_manager.update_file_metadata(metadata)
                self.queue.task_done()

    def _run_search(self, model_path: Optional[str], spectra_path: str, config_path: Optional[str],
                    metadata: SearchMetadata):
        """Run a search task (same as in original CasanovoDB)"""
        # Update status to 'running'
        metadata.status = 'running'
        # Preserve the owner_id when updating the metadata
        self.searches_manager.update_file_metadata(metadata)
        search_path = self.searches_manager.retrieve_file_path(
            metadata.file_id).strip(f'.{metadata.file_type}')

        # Command to run Casanovo
        command = [
            'casanovo',
            'sequence',
            spectra_path,
            '--output', search_path,
            '--verbosity', 'debug'
        ]

        if model_path:
            command.extend(['--model', model_path])

        if config_path:
            command.extend(['--config', config_path])

        try:
            # Execute the command
            subprocess.run(command, check=True)
            # Update status to 'completed' if successful
            metadata.status = 'completed'
        except subprocess.CalledProcessError:
            # Update status to 'failed' if there is an error
            metadata.status = 'failed'

        self.searches_manager.update_file_metadata(metadata)

    def _run_train(self, spectra_paths: list[str], config_path: Optional[str], metadata: ModelFileMetadata):
        """Run a training task (same as in original CasanovoDB)"""
        # Update status to 'running'
        metadata.status = 'running'
        self.models_manager.update_file_metadata(metadata)

        output_path = self.models_manager.retrieve_file_path(
            metadata.file_id).strip(f'.{metadata.file_type}')

        # Construct the command for running Casanovo
        command = [
            "casanovo", "sequence",
            *spectra_paths,
            "--output", output_path,
            "--verbosity", "debug"
        ]

        if config_path:
            command.extend(["--config", config_path])

        # Run the command
        try:
            subprocess.run(command, check=True)
            # Update status to 'completed' if successful
            metadata.status = 'completed'
        except subprocess.CalledProcessError:
            # Update status to 'failed' if there is an error
            metadata.status = 'failed'

        # Update the metadata with the final status
        self.models_manager.update_file_metadata(metadata)

    def _log(self, message: str):
        """Log a message if verbose mode is enabled"""
        if self.verbose:
            print(f'CasanovoAdvancedDB: {message}')

    def stop(self):
        """Stop the queue thread"""
        self.stop_event.set()
        self.queue.put(None)  # Ensure the thread exits the loop
        self.queue_thread.join()

    def train(self, spectra_ids: list[str], config_id: Optional[str], metadata: ModelFileMetadata,
              user_id: Optional[str] = None) -> str:
        """Train a model"""
        if user_id is None:
            user_id = self.default_user_id

        # Save model metadata to db
        self.models_manager.add_file(
            None, metadata, copy=False, owner_id=user_id)

        spectra_paths = [self.spectra_files_manager.retrieve_file_path(
            spectra_id) for spectra_id in spectra_ids]
        config_path = self.config_manager.retrieve_file_path(
            config_id) if config_id else None

        model_path = self.models_manager.retrieve_file_path(metadata.file_id)
        output_path = model_path.strip('.ckpt')

        self.queue.put({
            'target': 'train',
            'spectra_paths': spectra_paths,
            'config_path': config_path,
            'output_path': output_path,
            'metadata': metadata,
            'user_id': user_id
        })

        return metadata.file_id

    def search(self, metadata: SearchMetadata, config_id: Optional[str],
               user_id: Optional[str]) -> str:
        """Search for peptides"""

        if metadata.model is None:
            model_path = None
        else:
            model_path = self.models_manager.retrieve_file_path(metadata.model)

        self._log(f"Model path: {model_path}")
        spectra_path = self.spectra_files_manager.retrieve_file_path(
            metadata.spectra)
        self._log(f"Spectra path: {spectra_path}")
        search_id = metadata.file_id

        self._log(
            f"Running search with model: {model_path}, spectra: {spectra_path}, search_id: {search_id}")

        config_path = self.config_manager.retrieve_file_path(
            config_id) if config_id else None

        # Add initial metadata with 'pending' status, and file_path as None since it's not yet created
        self.searches_manager.add_file(
            None, metadata, copy=False, owner_id=user_id)

        # Add the search task to the queue
        self.queue.put({
            'target': 'search',
            'model_path': model_path,
            'spectra_path': spectra_path,
            'config_path': config_path,
            'metadata': metadata,
            'user_id': user_id
        })

        return search_id

    def get_queued_tasks(self) -> List[Dict]:
        """Get all queued tasks"""
        tasks = []
        for i in range(self.queue.qsize()):
            tasks.append(self.queue.queue[i])
        return tasks

    def edit_queued_task(self, index: int, new_model_id: Optional[str] = None, new_spectra_id: Optional[str] = None):
        """Edit a queued task"""
        if 0 <= index < self.queue.qsize():
            task = self.queue.queue[index]
            if new_model_id:
                task['model_id'] = self.models_manager.retrieve_file_path(
                    new_model_id)
            if new_spectra_id:
                task['spectra_id'] = self.spectra_files_manager.retrieve_file_path(
                    new_spectra_id)
            self.queue.queue[index] = task
        else:
            raise IndexError("Queue index out of range")

    def stop_queue(self):
        """Stop the queue"""
        self.queue.put(None)
        self.queue_thread.join()

    def reset_db(self):
        """Reset the database"""
        self.models_manager.reset_db()
        self.spectra_files_manager.reset_db()
        self.searches_manager.reset_db()
        self.config_manager.reset_db()

    def get_search_path(self, search_id: str):
        """Get the path to a search file"""
        path = self.searches_manager.retrieve_file_path(search_id)

        # Ensure it's completed
        search_metadata = self.searches_manager.get_file_metadata(search_id)

        if search_metadata.status != 'completed':
            warnings.warn(f"Search {search_id} is not completed yet.")
            return None

        return path

    def update_unfinished_searches(self):
        """Update all unfinished searches to 'failed' status"""
        for search_id in self.searches_manager.get_all_file_ids():
            try:
                search_metadata = self.searches_manager.get_file_metadata(
                    search_id)
                if search_metadata.status != 'completed':
                    search_metadata.status = 'failed'
                    self.searches_manager.update_file_metadata(search_metadata)
            except Exception as e:
                self._log(f"Error updating unfinished search {search_id}: {e}")

    def delete_search(self, search_id: str):
        """Delete a search and its log file"""
        search_path = self.searches_manager.retrieve_file_path(search_id)
        log_path = search_path.replace('.mztab', '.log')

        self.searches_manager.delete_file(search_id)

        if os.path.exists(log_path):
            os.remove(log_path)

    # User management methods
    def create_user(self, username: str, email: str, password: str, is_admin: bool = False) -> str:
        """Create a new user"""
        return self.user_manager.create_user(username, email, password, is_admin)

    def authenticate_user(self, username: str, password: str) -> Optional[str]:
        """Authenticate a user and return user ID if successful"""
        return self.user_manager.authenticate_user(username, password)

    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID"""
        return self.user_manager.get_user(user_id)

    def list_users(self) -> List[Dict[str, Any]]:
        """List all users"""
        return self.user_manager.list_users()

    # Filter methods by user
    def get_user_models(self, user_id: str) -> List[ModelFileMetadata]:
        """Get all models owned by a user"""
        return self.models_manager.get_all_metadata(user_id)

    def get_user_spectra(self, user_id: str) -> List[SpectraFileMetadata]:
        """Get all spectra files owned by a user"""
        return self.spectra_files_manager.get_all_metadata(user_id)

    def get_user_searches(self, user_id: str) -> List[SearchMetadata]:
        """Get all searches owned by a user"""
        return self.searches_manager.get_all_metadata(user_id)

    def get_user_configs(self, user_id: str) -> List[ConfigFileMetadata]:
        """Get all config files owned by a user"""
        return self.config_manager.get_all_metadata(user_id)


# Alias for backward compatibility
CasanovoDB = CasanovoAdvancedDB
