PROMPT = """Think step by step to carry out the instruction.
Instruction: Put the checkerboard round into the yellow and purple polka dot pan.
Program:
MASKS=SAM(image=IMAGE)
OBJS, MASKS=ImageCrop(image=IMAGE, masks=MASKS)
OBJ0=CLIPRetrieval(objs=OBJS, query='the yellow and purple polka dot pan')
LOC0=Pixel2Loc(obj=OBJ0, masks=MASKS)
OBJ1=CLIPRetrieval(objs=OBJS, query='the checkerboard round', obj0=OBJ0)
LOC1=Pixel2Loc(obj=OBJ1, masks=MASKS)
PickPlace(pick=LOC1, place=LOC0, bounds=BOUNDS)
EXE=ROBOT()

Instruction: Put the {dragged_obj} into the {base_obj}.
Program:
MASKS=SAM(image=IMAGE)
OBJS, MASKS=ImageCrop(image=IMAGE, masks=MASKS)
OBJ0=CLIPRetrieval(objs=OBJS, query=templates['base_obj'])
LOC0=Pixel2Loc(obj=OBJ0, masks=MASKS)
OBJ1=CLIPRetrieval(objs=OBJS, query=templates['dragged_obj'])
LOC1=Pixel2Loc(obj=OBJ1, masks=MASKS)
PickPlace(pick=LOC1, place=LOC0, bounds=BOUNDS)
EXE=ROBOT()

Instruction: Rotate the red letter R 150 degrees.
Program:
MASKS=SAM(image=IMAGE)
OBJS, MASKS=ImageCrop(image=IMAGE, masks=MASKS)
OBJ0=CLIPRetrieval(objs=OBJS, query='the red letter R')
LOC0=Pixel2Loc(obj=OBJ0, masks=MASKS)
PickPlace(pick=LOC0, place=LOC0, bounds=BOUNDS, yaw_angle=150, degrees=True)

Instruction: Rotate the magma hexagon 45 radius.
Program:
MASKS=SAM(image=IMAGE)
OBJS, MASKS=ImageCrop(image=IMAGE, masks=MASKS)
OBJ0=CLIPRetrieval(objs=OBJS, query='the magama hexagon')
LOC0=Pixel2Loc(obj=OBJ0, masks=MASKS)
PickPlace(pick=LOC0, place=LOC0, bounds=BOUNDS, yaw_angle=45, degrees=False)

Instruction: Put the {dragged_obj} into the {base_obj_1} then {base_obj_2}. Finally restore it into its original container..
Program:
MASKS=SAM(image=IMAGE) 
OBJS, MASKS=ImageCrop(image=IMAGE, masks=MASKS)
OBJ0=CLIPRetrieval(objs=OBJS, query=templates['base_obj_1']).format(query_1)
LOC0=Pixel2Loc(obj=OBJ0, masks=MASKS)
OBJ1=CLIPRetrieval(objs=OBJS, query=templates['base_obj_2'], pre_obj1=OBJ0).format(query_2)
LOC1=Pixel2Loc(obj=OBJ1, masks=MASKS)
OBJ2=CLIPRetrieval(objs=OBJS, query=templates['dragged_obj'], pre_obj1=OBJ0, pre_obj2=OBJ1).format(query_3)
LOC2=Pixel2Loc(obj=OBJ2, masks=MASKS)
PickPlace(pick=LOC2, place=LOC0, bounds=BOUNDS)
PickPlace(pick=LOC0, place=LOC1, bounds=BOUNDS)
PickPlace(pick=LOC1, place=LOC2, bounds=BOUNDS)

Instruction: Rearrange to this {scene}.
Program:
MASKS_OBS=SAM(image=IMAGE)
OBJS_OBS, MASKS_OBS=ImageCrop(image=IMAGE, masks=MASKS_OBS)
MASKS_GOAL=SAM(image=templates['scene'])
OBJS_GOAL, MASKS_GOAL=ImageCrop(image=templates['scene'], masks=MASKS_GOAL)
ROW, COL=get_objs_match(OBJS_GOAL, OBJS_OBS)
DistractorActions=DistractorActions(MASKS_OBS, COL, bounds=BOUNDS)
RearrangeActions=RearrangeActions(place_masks=MASKS_GOAL, pick_masks=MASKS_OBS, place_ind=ROW, pick_ind=COL, bounds=BOUNDS)


Instruction: Put the yellow and blue stripe object in scene into the rainbow object.
Program:
MASKS_OBS=SAM(image=IMAGE)
OBJS_OBS, MASKS_OBS=ImageCrop(image=IMAGE, masks=MASKS_OBS) 
MASKS_GOAL=SAM(image=templates['scene'])
OBJS_GOAL, MASKS_GOAL=ImageCrop(image=templates['scene'], masks=MASKS_GOAL)
GOAL=CLIPRetrieval(OBJS_GOAL, query='the yellow and blue stripe object')
TARGET=CLIPRetrieval(OBJS_OBS, query=OBJS_GOAL[GOAL])
LOC0=Pixel2Loc(TARGET, MASKS_OBS)
OBJ1=CLIPRetrieval(OBJS_OBS, query='the rainbow object', pre_obj=TARGET)
LOC1=Pixel2Loc(OBJ1, MASKS_OBS)
PickPlace(pick=LOC0, place=LOC1, bounds=BOUNDS)

Instruction: INSERT TASK HERE.
Program:
"""