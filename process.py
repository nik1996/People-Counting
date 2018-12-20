import subprocess
import matlab.engine

eng = matlab.engine.start_matlab()

eng.addp(nargout=0)
eng.vl_compilenn(nargout=0)

print("\nProcessing raw images...")

eng.Prep(nargout=0)
subprocess.call("python predict.py", shell=True)
eng.Evaluate(nargout=0)

print("Output generated successfully.")