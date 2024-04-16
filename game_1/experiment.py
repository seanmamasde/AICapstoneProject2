import subprocess
import sys
import os
import time
import psutil


def run_game(executable, num_times):
    results = []

    for _ in range(num_times):
        # Start the process without creating a new command line window
        if os.path.exists('temp_output.txt'):
            os.remove('temp_output.txt')
        with open('temp_output.txt', 'w+') as f:
            process = subprocess.Popen(executable, stdout=f, stderr=f, text=True,
                                       creationflags=subprocess.CREATE_NO_WINDOW)

        # Wait for the expected game over indication or a fixed amount of time
        time.sleep(60)  # Adjust this based on typical game length

        # Force quit the AI_game.exe process if it's still running
        if process.poll() is None:  # Check if the process has not ended
            for proc in psutil.process_iter(['pid', 'name']):
                if 'AI_game.exe' in proc.info['name']:
                    proc.terminate()
                    print(f"Process {proc.info['name']} terminated.")

        # Wait for the process to end
        process.wait()

        # Read the output from the file
        with open('temp_output.txt', 'r') as f:
            lines = f.readlines()

        game_over = False
        score_board = []
        winner = None

        for line in lines:
            if '----------END GAME----------' in line:
                game_over = True
            elif game_over and 'Winner' in line:
                winner = line.strip()
            elif game_over and 'Score Board' not in line and 'Press any key to continue' not in line and line.strip():
                score_board.append(line.strip())

        results.append((score_board, winner))

    return results


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python script.py <path_to_executable> <num_times>")
        sys.exit(1)

    executable_path = sys.argv[1]
    num_times = int(sys.argv[2])
    game_results = run_game(executable_path, num_times)

    for i, result in enumerate(game_results):
        score_board, winner = result
        print(f"Game {i + 1} Results:")
        for score in score_board:
            print(score)
        print(winner)
        print()
