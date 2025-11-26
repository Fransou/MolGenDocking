from typing import Any, Dict, List, Tuple


class Verifier:
    def __init__(self) -> None:
        pass

    def get_score(
        self, completions: List[Any], metadata: List[Dict[str, Any]]
    ) -> Tuple[List[float], List[Dict[str, Any]]]:
        raise NotImplementedError
