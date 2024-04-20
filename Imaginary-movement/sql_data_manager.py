import sqlite3 as sql
import torch.utils.data as torch_data
import torch
import numpy.typing as npt
import numpy as np
from tqdm import tqdm

class SqlDataSet(torch_data.Dataset):
    def __init__(self, database_path: str) -> None:
        super().__init__()
        self.connection = sql.connect(database_path)
        self.path = database_path
        self.create_tables()

    def create_tables(self):
        c = self.connection.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS data(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_length INTEGER NOT NULL,
                signals BINARY,
                labels BINARY
            );
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS meta(
                title,
                channel_names
            );
        """)

        self.connection.commit()

    def __len__(self) -> int:
        r = self.connection.execute("SELECT COUNT (*) from data")
        return r.fetchone()[0]

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns a single entry as a batch of 1 item.
        
        """
        r = self.connection.execute("SELECT signal_length, signals, labels FROM data WHERE id = ?", (index+1,))
        count, signals, labels = r.fetchone()
        
        # signals are 64 pixel frames with 1 channel each (Can also be treated as a 1d strip with 1 channel)
        signals = (np.frombuffer(signals, dtype='<u2').astype('float32') / 65_535.0).reshape((count, 8, 8))
        labels = (np.frombuffer(labels, dtype='<u1').astype('float32') / 9.0).reshape((count, 1))
        
        return (torch.from_numpy(signals), torch.from_numpy(labels))

    def get_signal_length(self, index: int) -> int:
        r = self.connection.execute("SELECT signal_length FROM data WHERE id = ?", (index+1,))
        signal_length = r.fetchone()
        return signal_length[0]

    def add_data(self, signals: npt.NDArray[np.uint16] , labels: npt.NDArray[np.uint8]):
        #The data should be on the form (signals, channels)
        
        if signals.shape[0] == 64:
            signals = signals.transpose()

        if signals.shape[1] != 64:
            raise BaseException("wrong number of signals")
        
        signal_bytes = sql.Binary(signals.astype('<u2').tobytes())
        label_bytes = sql.Binary(labels.astype('<u1').tobytes())

        c = self.connection.cursor()
        c.execute("INSERT INTO data(signal_length, signals, labels) VALUES(?, ?, ?)", (signals.shape[0], signal_bytes, label_bytes))
        self.connection.commit()

    def close_db(self):
        self.connection.close()
    
    def open_db(self):
        self.connection = sql.connect(self.path)
    
class ChunkedEdfDatabase(torch_data.Dataset):
    def __init__(self, source: SqlDataSet, chunk_size: int, /, seed: int | None = None, max_allowed_overlap: int | None = None, max_number_of_chunks: int | None = None) -> None:
        super().__init__()
        self.source = source
        self.chunks = []
        self.chunk_size = chunk_size
        self.rnd = np.random.default_rng(seed=seed)
        if max_allowed_overlap == None:
            self.max_allowed_overlap = chunk_size // 2
        else:
            self.max_allowed_overlap = max_allowed_overlap
        
        self.max_number_of_chunks = max_number_of_chunks

        self.split_series()
        
    def get_random_overlap(self):
        return int(self.max_allowed_overlap * self.rnd.random())

    def split_series(self):
        chunks = []
        for i in tqdm(range(len(self.source))):
            l = self.source.get_signal_length(i)
            pos = 0
            while pos < l - self.chunk_size:
                chunks.append({
                    'index': i,
                    'start': pos,
                    'end': pos+self.chunk_size
                })
                pos += self.get_random_overlap()
        
        self.rnd.shuffle(chunks)
        if self.max_number_of_chunks is not None:
            chunks = chunks[:self.max_number_of_chunks]
        self.chunks = chunks
        print(f"created {len(chunks)} chunks from {len(self.source)} entries")

    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.chunks[index]
        (signals, labels) = self.source[chunk['index']]

        signals = signals[:,:,:,chunk['start']:chunk['end']]
        labels = labels[:,chunk['start']:chunk['end']]

        return (signals, labels)

    def connect(self):
        self.source.open_db()

    def close(self):
        self.source.close_db()