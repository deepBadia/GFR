import yaml
import h5py
from surmod.main import train_model, generate_results
from surmod.utils.common import (
        get_project_root,
        resolve_data_path,
        load_model_module,
        load_checkpoint,
        evaluate_model,
        Normalizer,
        compute_gain_phase_rmse,
    )
from surmod.core.data_loader import DataLoader as GFRDataLoader

def _runDataInfo(folder:str,filename:str)-> None:
    def _explorer_hdf5(name, obj):
        print("\n" + "=" * 80)
        if isinstance(obj, h5py.Dataset):
            print("Type   : Dataset")
            print(f"Name DataSet: {obj.name}")
            print(f"Shape  : {obj.shape}")
            print(f"Dtype  : {obj.dtype}")
            # Affichage des attributs
            if len(obj.attrs) > 0:
                print("\nAttributs :")
                for key, value in obj.attrs.items():
                    print(f"  {key} = {value}")
            try:
                data = obj[...]

                print("\nData overview (first Line ):")
                print(data[0, :])

            except Exception as e:
                print(f"Impossible to read the dataSet, verify your .h file: {e}")
        return data

    with h5py.File(folder / filename, "r") as f:

        print("\nFULL CONTENT of HDF5 file : ", filename)
        print("=" * 80)

        data = f.visititems(_explorer_hdf5)

if __name__ == "__main__":
    if 0 :
        #################################
        # GRF
        #################################
        RunTrain = 1
        #################################
        # Lecture data
        #################################
        root = get_project_root()
        data_dir = root / "data"
        filename = "secteur1.h5"
        print(f"Project root : {root}")
        print(f"Data folder  : {data_dir}")
        _runDataInfo(data_dir, filename)
        config_path = root / "../IA_mesure/configsSecteur1.yaml"
        results_root = root / "../results"
        results_root.mkdir(parents=True, exist_ok=True)
        # ------------------------------------------------------------------
        # 3. Train
        # ------------------------------------------------------------------

        if RunTrain == 1:
            best_loss = train_model(
                experiment="comp_full_gfr_Secteur1",
                config_path=str(config_path),
                save_path=str(root / ".."),
                tuner=False,
            )
            print(f"\nBest validation loss: {best_loss:.6e}")
        pth_files = sorted(results_root.rglob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)

        if not pth_files:
            raise FileNotFoundError("No checkpoint found under results/")

        run_dir = pth_files[0].parent
        model_path = pth_files[0]
        print(f"Generating results for: {run_dir}")

        generate_results(run_dir)


    if 1 :
        #################################
        # GFR active learning
        #################################
        RunTrain = 1
        #################################
        # Lecture data
        #################################
        root = get_project_root()
        data_dir = root / "data"
        filename = "secteur1.h5"
        print(f"Project root : {root}")
        print(f"Data folder  : {data_dir}")
        _runDataInfo(data_dir, filename)
        config_path = root / "../IA_mesure/configsSecteur1.yaml"
        results_root = root / "../results"
        results_root.mkdir(parents=True, exist_ok=True)
        # ------------------------------------------------------------------
        # 3. Train
        # ------------------------------------------------------------------

        if RunTrain == 1:
            best_loss = train_model(
                experiment="comp_activelearning_gfr_Secteur1",
                config_path=str(config_path),
                save_path=str(root / ".."),
                tuner=False,
            )
            print(f"\nBest validation loss: {best_loss:.6e}")
        pth_files = sorted(results_root.rglob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)

        if not pth_files:
            raise FileNotFoundError("No checkpoint found under results/")

        run_dir = pth_files[0].parent
        model_path = pth_files[0]
        print(f"Generating results for: {run_dir}")

        generate_results(run_dir)




