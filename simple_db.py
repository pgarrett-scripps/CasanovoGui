import json
import os
import queue
import shutil
import subprocess
import threading
import uuid
import warnings
from typing import Type, TypeVar, Generic, List, Optional, Literal, Dict

from tinydb import TinyDB, Query
from pydantic import BaseModel
from datetime import datetime
from datetime import date

# Define a generic type variable for the Pydantic model
T = TypeVar('T', bound=BaseModel)


class _TinyDBManager(Generic[T]):
    def __init__(self,
                 model: Type[T],
                 db_path: str,
                 table_name: str,
                 lock: threading.Lock,
                 verbose: bool,
                 ):
        self.model = model
        self.db = TinyDB(db_path)
        self.files_table = self.db.table(table_name)
        self.verbose = verbose
        self.lock = lock

    def _log(self, message: str):
        if self.verbose:
            print(f'TinyDBManager[{self.model.__name__}]: {message}')

    def add_metadata(self, metadata: T):
        data = metadata.dict()
        data['date'] = data['date'].isoformat()  # Convert date to string
        with self.lock:
            self.files_table.insert(data)
        self._log(f"Added metadata: {data}")

    def update_metadata(self, metadata: T):
        File = Query()
        data = metadata.dict()
        data['date'] = data['date'].isoformat()  # Convert date to string
        with self.lock:
            self.files_table.update(data, File.file_id == metadata.file_id)
        self._log(f"Updated metadata: {data}")

    def delete_metadata(self, file_id: str):
        File = Query()
        with self.lock:
            self.files_table.remove(File.file_id == file_id)
        self._log(f"Deleted metadata for file_id: {file_id}")

    def get_metadata(self, file_id: str, model: Type[T]) -> Optional[T]:
        File = Query()
        with self.lock:
            result = self.files_table.search(File.file_id == file_id)
        if result:
            data = result[0]
            self._log(f"Retrieved raw data: {data}")
            if isinstance(data['date'], str):
                data['date'] = datetime.fromisoformat(data['date'])  # Convert string to date
            self._log(f"Converted data: {data}")
            return model(**data)
        self._log(f"No metadata found for file_id: {file_id}")
        return None


class FileManager(Generic[T]):
    def __init__(self,
                 model: Type[T],
                 storage_dir: str,
                 db_path: str,
                 table_name: str,
                 lock: threading.Lock,
                 verbose: bool = False,
                 ):
        self.storage_dir = storage_dir
        self.db_manager = _TinyDBManager[T](model, db_path, table_name, lock, verbose)
        self.model = model
        self.verbose = verbose
        os.makedirs(self.storage_dir, exist_ok=True)
        self.lock = threading.Lock()

    def _log(self, message: str):
        if self.verbose:
            print(f'FileManager[{self.model.__name__}]: {message}')

    def add_file(self, file_path: str, metadata: T, copy: bool = True) -> str:
        file_id = metadata.file_id
        dest_path = os.path.join(self.storage_dir, file_id + '.' + metadata.file_type)

        if os.path.exists(dest_path):
            raise FileExistsError(f"File '{file_id}' already exists in storage.")

        with self.lock:
            if file_path is not None:
                if copy:
                    shutil.copy(file_path, dest_path)
                else:
                    shutil.move(file_path, dest_path)

            metadata.file_id = file_id
            self.db_manager.add_metadata(metadata)
        self._log(f"Added file '{file_path}' as '{file_id}' with metadata {metadata}")
        return file_id

    def update_file_metadata(self, metadata: T) -> None:
        with self.lock:
            if not self.db_manager.get_metadata(metadata.file_id, self.model):
                raise FileNotFoundError(f"File metadata for '{metadata.file_id}' not found.")
            self.db_manager.update_metadata(metadata)
        self._log(f"Updated metadata for file '{metadata.file_id}'")

    def delete_file(self, file_id: str) -> None:
        with self.lock:
            file_path = os.path.join(self.storage_dir, file_id + f'.{self.get_file_metadata(file_id).file_type}')
            if not os.path.exists(file_path):
                warnings.warn(f"File '{file_id}' not found in storage.")
            else:
                os.remove(file_path)
            self.db_manager.delete_metadata(file_id)
        self._log(f"Deleted file '{file_id}' and its metadata")

    def get_file_metadata(self, file_id: str) -> T:
        metadata = self.db_manager.get_metadata(file_id, self.model)
        if not metadata:
            raise FileNotFoundError(f"File metadata for '{file_id}' not found.")
        self._log(f"Retrieved metadata for file '{file_id}': {metadata}")
        return metadata

    def get_all_metadata(self) -> List[T]:
        metadata_list = []
        with self.lock:
            for data in self.db_manager.files_table.all():
                if isinstance(data['date'], str):
                    data['date'] = datetime.fromisoformat(data['date'])
                metadata_list.append(self.model(**data))
        self._log(f"Retrieved all metadata: {metadata_list}")
        return metadata_list

    def get_all_file_ids(self) -> List[str]:
        with self.lock:
            return [data['file_id'] for data in self.db_manager.files_table.all()]

    def retrieve_file(self, file_id: str, dest_path: str) -> str:
        with self.lock:
            src_path = os.path.join(self.storage_dir, file_id)
            if not os.path.exists(src_path):
                raise FileNotFoundError(f"File '{file_id}' not found in storage.")
            shutil.copy(src_path, dest_path)
        self._log(f"Retrieved file '{file_id}' to '{dest_path}'")
        return dest_path

    def retrieve_file_path(self, file_id: str) -> str:
        with self.lock:
            src_path = os.path.join(self.storage_dir, file_id + f'.{self.get_file_metadata(file_id).file_type}')
            if not os.path.exists(src_path):
                raise FileNotFoundError(f"File '{file_id}' not found in storage.")
        return src_path

    def reset_db(self):
        with self.lock:
            # if files is a table, drop it
            if 'files' in self.db_manager.db.tables():
                self.db_manager.db.drop_table('files')

            for file in os.listdir(self.storage_dir):
                os.remove(os.path.join(self.storage_dir, file))

        self._log("Reset the database")


# Define different metadata models
class GeneralFileMetadata(BaseModel):
    file_id: str
    file_name: str
    description: str
    file_type: str
    date: date
    tags: List[str]


class ModelFileMetadata(GeneralFileMetadata):
    pass


class SpectraFileMetadata(GeneralFileMetadata):
    enzyme: str
    instrument: str


class SearchMetadata(GeneralFileMetadata):
    model_id: Optional[str]
    spectra_id: Optional[str]
    status: str = Literal['pending', 'running', 'completed', 'failed']


class CasanovoDB:
    """
    This sets up 3 file managers for models, files, and searches.
    """

    def __init__(self,
                 data_folder: str,
                 models_storage_folder: str = 'models',
                 models_table_name: str = 'models',
                 spectra_files_storage_folder: str = 'files',
                 spectra_files_table_name: str = 'files',
                 searches_storage_folder: str = 'searches',
                 searches_table_name: str = 'searches',
                 verbose: bool = False):

        assert len(
            set([models_table_name, spectra_files_table_name, searches_table_name])) == 3, "Table names must be unique"
        assert len(set([models_storage_folder, spectra_files_storage_folder,
                        searches_storage_folder])) == 3, "Storage folders must be unique"

        # if storage folder does not exist, create it
        os.makedirs(data_folder, exist_ok=True)

        self.metadata_db_path = os.path.join(data_folder, 'metadata_db.json')
        models_storage_folder = os.path.join(data_folder, models_storage_folder, )
        spectra_files_storage_folder = os.path.join(data_folder, spectra_files_storage_folder)
        searches_storage_folder = os.path.join(data_folder, searches_storage_folder)

        meta_data_lock = threading.Lock()

        self.models_manager = FileManager[ModelFileMetadata](ModelFileMetadata,
                                                             models_storage_folder,
                                                             self.metadata_db_path,
                                                             models_table_name,
                                                             meta_data_lock,
                                                             verbose,
                                                             )
        self.spectra_files_manager = FileManager[SpectraFileMetadata](SpectraFileMetadata,
                                                                      spectra_files_storage_folder,
                                                                      self.metadata_db_path,
                                                                      spectra_files_table_name,
                                                                      meta_data_lock,
                                                                      verbose,
                                                                      )
        self.searches_manager = FileManager[SearchMetadata](SearchMetadata,
                                                            searches_storage_folder,
                                                            self.metadata_db_path,
                                                            searches_table_name,
                                                            meta_data_lock,
                                                            verbose,
                                                            )
        self.verbose = verbose

        self.queue = queue.Queue()
        self.current_task = None
        self.queue_thread = threading.Thread(target=self._process_queue)
        self.queue_thread.daemon = False  # Set to True to stop the thread when the main thread exits
        self.queue_thread.start()

        self.update_unfinished_searches()

    def _process_queue(self):
        while True:
            task = self.queue.get()
            if task is None:
                break
            try:
                self._run_search(task['model_id'], task['spectra_id'], task['search_id'], task['search_metadata'])
            except Exception as e:
                print(f"Error in search: {e}")
            self.queue.task_done()

    def _run_search(self, model_path: str, spectra_path: str, output_path: str, search_metadata: SearchMetadata):
        # Update status to 'running'
        search_metadata.status = 'running'
        self.searches_manager.update_file_metadata(search_metadata)
        final_result_path = os.path.join(self.searches_manager.storage_dir, output_path)

        # Command to run Casanovo
        command = [
            'casanovo',
            'sequence',
            spectra_path,
            '--output', final_result_path,
            '--verbosity', 'info'
        ]

        if model_path is not None:
            command.extend(['--model', model_path])

        # Execute the command
        res = subprocess.run(command, check=False)

        if res.returncode != 0:
            search_metadata.status = 'failed'
        else:
            search_metadata.status = 'completed'

        self.searches_manager.update_file_metadata(search_metadata)

        """            
        Causes issues with logger.... cant fix it :(
        
        output = setup_logging(final_result_path, 'info')
        logger = logging.getLogger("casanovo")
        config, model = setup_model(model_id, None, output, False)
        with ModelRunner(config, model) as runner:
            logger.info("Sequencing peptides from:")
            for peak_file in [spectra_id]:
                logger.info("  %s", peak_file)
            runner.predict([spectra_id], output)
        logger.info("DONE!")
        """

    def _log(self, message: str):
        if self.verbose:
            print(f'CasanovoDB: {message}')

    def search(self, metadata: SearchMetadata) -> str:
        if metadata.model_id is None:
            model_path = None
        else:
            model_path = self.models_manager.retrieve_file_path(metadata.model_id)

        self._log(f"Model path: {model_path}")
        spectra_path = self.spectra_files_manager.retrieve_file_path(metadata.spectra_id)
        self._log(f"Spectra path: {spectra_path}")
        search_id = metadata.file_id

        self._log(f"Running search with model: {model_path}, spectra: {spectra_path}, search_id: {search_id}")

        # Add initial metadata with 'pending' status, and file_path as None since it's not yet created
        self.searches_manager.add_file(None, metadata, copy=False)

        # Add the search task to the queue
        self.queue.put({
            'model_id': model_path,
            'spectra_id': spectra_path,
            'search_id': search_id,
            'search_metadata': metadata
        })

        return search_id

    def get_queued_tasks(self) -> List[Dict]:
        tasks = []
        for i in range(self.queue.qsize()):
            tasks.append(self.queue.queue[i])
        return tasks

    def edit_queued_task(self, index: int, new_model_id: Optional[str] = None, new_spectra_id: Optional[str] = None):
        if 0 <= index < self.queue.qsize():
            task = self.queue.queue[index]
            if new_model_id:
                task['model_id'] = self.models_manager.retrieve_file_path(new_model_id)
            if new_spectra_id:
                task['spectra_id'] = self.spectra_files_manager.retrieve_file_path(new_spectra_id)
            self.queue.queue[index] = task
        else:
            raise IndexError("Queue index out of range")

    def stop_queue(self):
        self.queue.put(None)
        self.queue_thread.join()

    def reset_db(self):
        try:
            self.models_manager.reset_db()
            self.spectra_files_manager.reset_db()
            self.searches_manager.reset_db()
        except json.decoder.JSONDecodeError:
            # remove metadata_db.json if it is corrupted
            os.remove(self.metadata_db_path)
            self.models_manager.reset_db()
            self.spectra_files_manager.reset_db()
            self.searches_manager.reset_db()

    def get_search_path(self, search_id: str):
        path = self.searches_manager.retrieve_file_path(search_id)

        # ensure its completed
        search_metadata = self.searches_manager.get_file_metadata(search_id)

        if search_metadata.status != 'completed':
            warnings.warn(f"Search {search_id} is not completed yet.")
            return None

        return path

    def update_unfinished_searches(self):
        for search_id in self.searches_manager.get_all_file_ids():
            search_metadata = self.searches_manager.get_file_metadata(search_id)
            if search_metadata.status == 'running':
                search_metadata.status = 'failed'
                self.searches_manager.update_file_metadata(search_metadata)

    def delete_search(self, search_id: str):
        # delete search and log file
        search_metadata = self.searches_manager.get_file_metadata(search_id)
        search_path = self.searches_manager.retrieve_file_path(search_id)

        log_path = search_path.replace('.mztab', '.log')

        self.searches_manager.delete_file(search_id)

        if os.path.exists(log_path):
            os.remove(log_path)


# Usage Example
if __name__ == "__main__":
    from datetime import date

    # Initialize the CasanovoDB
    db = CasanovoDB('file_storage', verbose=True)

    db.reset_db()

    spectra_file_path = r'C:\Users\Ty\PycharmProjects\CasanovoGui\sample_preprocessed_spectra.mgf'
    model_file_path = r'C:\Users\Ty\PycharmProjects\CasanovoGui\casanovo_nontryptic.ckpt'

    # Add a model file
    model_metadata = ModelFileMetadata(
        file_id=str(uuid.uuid4()),
        file_name='model_1',
        description='Casanovo model',
        file_type='ckpt',
        date=date.today(),
        tags=['model', 'casanovo']
    )

    model_id = db.models_manager.add_file(model_file_path, model_metadata)
    print(f"Model file saved as: {model_id}")

    # Add a spectra file
    spectra_metadata = SpectraFileMetadata(
        file_id=str(uuid.uuid4()),
        file_name='spectra_1',
        description='Preprocessed spectra',
        file_type='mgf',
        date=date.today(),
        tags=['spectra', 'preprocessed'],
        enzyme='trypsin',
        instrument='Orbitrap'
    )

    spectra_id = db.spectra_files_manager.add_file(spectra_file_path, spectra_metadata)
    print(f"Spectra file saved as: {spectra_id}")

    for i in range(2):
        # Initial search metadata
        search_metadata = SearchMetadata(
            file_id=str(uuid.uuid4()),
            file_name="Simple Search",
            description='Search results',
            file_type='mztab',
            date=date.today(),
            tags=['search', 'results'],
            model_id=model_id,
            spectra_id=spectra_id,
            status='pending'
        )

        print(f"Search metadata: {search_metadata}")

        # Perform a search
        search_id = db.search(search_metadata)
        print(f"Search results saved as: {search_id}")

    # Wait for the queue to be processed
    db.queue.join()
    print("All searches completed.")


