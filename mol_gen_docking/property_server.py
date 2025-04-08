import pandas as pd

from fastapi import FastAPI, BackgroundTasks, Query

from mol_gen_docking.utils.molecular_properties import get_oracle, KNOWN_PROPERTIES


app = FastAPI()


class PropertyServer:
    def __init__(self, max_cache_size_per_property: int = 10**4, rescale: bool = True):
        self.oracles = {oracle: get_oracle(oracle) for oracle in KNOWN_PROPERTIES}
        self.smiles_cache = {
            oracle: pd.DataFrame(columns=["smiles", "property"]).set_index("smiles")
            for oracle in KNOWN_PROPERTIES
        }
        self.max_cache_size_per_property = max_cache_size_per_property
        self.rescale = rescale

    def compute_properties(self, smiles: list[str], prop: str) -> list[float]:
        """
        Compute the properties of a molecule.
        :param smiles: SMILES of the molecule
        :return: Dictionary of the properties
        """
        propeties = []
        for s in smiles:
            if s not in self.smiles_cache[prop]:
                mol_prop = float(self.oracles[prop](s, rescale=self.rescale)[0])
                propeties.append(mol_prop)
            else:
                propeties.append(self.smiles_cache[prop].loc[s])
        return propeties

    def query_property(self, smiles_list: list[str], prop: str) -> None:
        """
        Query the properties of a molecule.
        :param smiles: SMILES of the molecule
        :return: None
        """
        smiles_to_add = [s for s in smiles_list if s not in self.smiles_cache[prop]]
        n_to_remove = len(smiles_to_add) - (
            self.max_cache_size_per_property - len(self.smiles_cache[prop])
        )
        if n_to_remove > 0:
            idx_to_remove = []
            for i, s in enumerate(self.smiles_cache[prop].index):
                if s not in smiles_list:
                    idx_to_remove.append(i)
                if len(idx_to_remove) == n_to_remove:
                    break

            self.smiles_cache[prop] = self.smiles_cache[prop].drop(
                self.smiles_cache[prop].index[idx_to_remove]
            )

        mol_prop = self.oracles[prop](smiles_to_add, rescale=self.rescale)
        for i, (smiles, p) in enumerate(zip(smiles_to_add, mol_prop)):
            self.smiles_cache[prop].loc[smiles] = mol_prop[i]


property_server = PropertyServer()


@app.get("/property/{property}/")
def compute_properties(property: str, smiles: list[str] | None = Query(None)):
    if smiles is None:
        return {"message": "No SMILES provided."}
    prop = property_server.compute_properties(smiles, property)
    return {"property": prop, "smiles": smiles}


@app.post("/property/{property}/")
async def query_property(
    property: str,
    background_tasks: BackgroundTasks,
    smiles: list[str] | None = Query(None),
):
    background_tasks.add_task(property_server.query_property, smiles, property)
    return {"message": "Computing properties in the background."}
