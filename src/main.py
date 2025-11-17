import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eda
import preprocess
import train
import hypertuning

def print_menu():
    print("\n" + "=" * 60)
    print("Machine Learning Pipeline - Student Success Prediction")
    print("=" * 60)
    print("\nPlease select an option:")
    print("  1. Exploratory Data Analysis (EDA)")
    print("  2. Data Preprocessing")
    print("  3. Model Training")
    print("  4. Hyperparameter Tuning & Feature Engineering")
    print("  5. Run Full Workflow (All Steps)")
    print("  0. Exit")
    print("=" * 60)

def run_full_workflow():
    print("\n" + "=" * 60)
    print("Running Full Workflow")
    print("=" * 60)
    
    try:
        print("\n>>> Step 1/4: Exploratory Data Analysis")
        eda.run_eda()
        
        print("\n>>> Step 2/4: Data Preprocessing")
        preprocess.run_preprocessing()
        
        print("\n>>> Step 3/4: Model Training")
        train.run_training()
        
        print("\n>>> Step 4/4: Hyperparameter Tuning & Feature Engineering")
        hypertuning.run_hypertuning()
        
        print("\n" + "=" * 60)
        print("Full Workflow Completed Successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError in full workflow: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    while True:
        print_menu()
        choice = None
        
        try:
            choice = input("\nEnter your choice: ").strip()
            
            if choice == '0':
                print("\nExiting... Goodbye!")
                break
            elif choice == '1':
                print("\n>>> Running Exploratory Data Analysis...")
                eda.run_eda()
            elif choice == '2':
                print("\n>>> Running Data Preprocessing...")
                preprocess.run_preprocessing()
            elif choice == '3':
                print("\n>>> Running Model Training...")
                train.run_training()
            elif choice == '4':
                print("\n>>> Running Hyperparameter Tuning & Feature Engineering...")
                print("WARNING: Hyperparameter tuning & feature engineering can take a long time (several minutes to hours).")
                proceed = input("Do you wish to proceed? (y/n): ").strip().lower()
                if proceed != 'y':
                    print("Hyperparameter tuning cancelled by user.")
                else:
                    try:
                        n_iter = input("Enter number of iterations for hyperparameter search (default: 50): ").strip()
                        n_iter = int(n_iter) if n_iter else 50
                        hypertuning.run_hypertuning(n_iter=n_iter)
                    except ValueError:
                        print("Invalid input. Using default 50 iterations.")
                        hypertuning.run_hypertuning()
            elif choice == '5':
                run_full_workflow()
            else:
                print("\nInvalid choice. Please enter a number between 0 and 5.")
                choice = None
                
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user. Exiting...")
            break
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            import traceback
            traceback.print_exc()
            choice = None
        
        if choice and choice != '0':
            continue_choice = input("\nPress Enter to return to menu, or 'q' to quit: ").strip().lower()
            if continue_choice == 'q':
                print("\nExiting... Goodbye!")
                break

if __name__ == "__main__":
    main()