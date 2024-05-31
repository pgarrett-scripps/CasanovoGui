import uuid
from datetime import date

from casanovogui.simple_db import CasanovoDB, SpectraFileMetadata, SearchMetadata


def main():
    # Initialize the CasanovoDB
    db = CasanovoDB('file_storage', verbose=True)

    # Reset the database
    db.reset_db()

    spectra_file_path = r'./data/sample_preprocessed_2spectra.mgf'

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

    try:
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
                model_id=None,
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
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
