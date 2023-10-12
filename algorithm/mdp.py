import mdptoolbox.mdp as mdp
import mdptoolbox.example

P, R = mdptoolbox.example.forest(S=4)
vi = mdptoolbox.mdp.PolicyIteration(P, R, 0.9, policy0=[1,0,0,0])
vi = mdptoolbox.mdp.PolicyIterationModified(P, R, 0.9, policy0=[1,0,0,0])
vi = mdptoolbox.mdp.QLearning(P, R, 0.9, policy0=[1,0,0,0])
vi.run()
print(vi.policy)
print(vi.iter)
