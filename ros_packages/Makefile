BUILD=colcon build --packages-select

all : services in_hand_control

in_hand_control : in_hand_control.target
	touch $^
	${BUILD} $@

services : services.target
	touch $^
	${BUILD} $@
