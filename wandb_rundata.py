import wandb
api = wandb.Api()
run = api.run("/akul-tyagi-cs/fprm-atwd/runs/dq374cdf")

print(run.history())