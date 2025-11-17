# Practical session to manage file paths using dataclasses and pathlib

from pathlib import Path
from dataclasses import dataclass

@dataclass
class PathManager:
    base: Path

    def ensure(self, *paths: str) -> bool:
        try:
            full_path = self.base / Path(*paths)
            if not full_path.exists():
                print(f"Creating missing path: {full_path}")
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.touch()
            return True
        except Exception as e:
            print(f"Error ensuring path {full_path}: {e}")
            return False

def main():
    pm = PathManager(base=Path('path_manager/workspace'))

    data = pm.ensure('data', 'inputs.txt')
    print(f"Data path ensured: {data}")

    logs = pm.ensure('logs', 'session2.log')
    print(f"Logs path ensured: {logs}")

if __name__ == "__main__":
    main()