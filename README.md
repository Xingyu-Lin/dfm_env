# dfm_env
Environments for manipulating deformable objects

# Run

Currently only rope.py is working

# Notes for creating new environment
* Any new environment should inherit from the base class
* Methods that need to be implement
    * _reset_sim: Reset the simulation and should indicate whether the reset is successful
    * _get_obs
    * _set_action
    * \[optional\] get_current_info
    * \[optional\] set_hidden_goal