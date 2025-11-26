# -*- coding: utf-8 -*-

import time
import queue

from ._sets_units import (
    derivedArgSet, renderOnlyArgSet, compoundArgs, diagnosticArgs)

from .beamline import BeamLine


def propagationProcess(q_in, q_out):
    handler = MessageHandler()
    repeats = 0
    while True:
        try:
            message = q_in.get_nowait()
#            print("MH", message)
            handler.process_message(message)
        except queue.Empty:
            pass
#            time.sleep(0.1)
        if handler.exit:
            break

        if handler.stop:
            time.sleep(0.01)
#            continue
        elif handler.needUpdate:
            started = True if handler.startEl is None else False
            flowLen = len(handler.bl.flowU)
            flowCounter = 0
            for oeid, meth in handler.bl.flowU.items():
                if not started:  # Skip until the modified element
                    if handler.startEl == oeid:
                        started = True
                    else:
                        continue
                oe = handler.bl.oesDict[oeid][0]

                for func, fkwargs in meth.items():
                    try:
                        getattr(oe, func)(**fkwargs)
                    except Exception as e:
                        raise
                        print("Error in PropagationProcess\n", e)
                        continue
                    flowCounter += 1
                    for autoAttr in derivedArgSet:
                        # 'center', 'pitch', 'bragg', 'R', 'r', 'Rm', 'Rs'
                        if (hasattr(oe, f'_{autoAttr}') and hasattr(
                                oe, f'_{autoAttr}Val')):
                            if True:  # getattr(oe, f'_{autoAttr}') != getattr(
                                # oe, f'_{autoAttr}Val'):
                                msg_autopos_update = {
                                        'pos_attr': autoAttr,
                                        'pos_value': getattr(oe, autoAttr),
                                        'sender_name': oe.name,
                                        'sender_id': oe.uuid,
                                        'status': 0}
                                q_out.put(msg_autopos_update)

                    for autoAttr in ['footprint']:
                        if (hasattr(oe, autoAttr) and len(
                                getattr(oe, autoAttr)) > 0):
                            msg_autopos_update = {
                                    'pos_attr': autoAttr,
                                    'pos_value': getattr(oe, autoAttr),
                                    'sender_name': oe.name,
                                    'sender_id': oe.uuid,
                                    'status': 0}
                            q_out.put(msg_autopos_update)

                    for diagAttrName in diagnosticArgs:
                        if hasattr(oe, diagAttrName):
                            diagAttrValue = getattr(oe, diagAttrName, None)
                            if diagAttrValue is not None:
                                msg_diagparam_update = {
                                        'diag_attr': diagAttrName,
                                        'diag_value': diagAttrValue,
                                        'sender_name': oe.name,
                                        'sender_id': oe.uuid,
                                        'status': 0}
                                q_out.put(msg_diagparam_update)

#                    for dependAttrName in dependentArgs:
#                        if hasattr(oe, dependAttrName):
#                            dependAttrValue = getattr(oe, dependAttrName, None)
#                            if dependAttrValue is not None:
#                                msg_dependparam_update = {
#                                        'depend_attr': dependAttrName,
#                                        'depend_value': dependAttrValue,
#                                        'sender_name': oe.name,
#                                        'sender_id': oe.uuid,
#                                        'status': 0}
#                                q_out.put(msg_dependparam_update)

                    msg_beam = {'beam': handler.bl.beamsDictU[oe.uuid],
                                'sender_name': oe.name,
                                'sender_id': oe.uuid,
                                'status': 0}
                    q_out.put(msg_beam)
                    # TODO: histDict
                    if hasattr(oe, 'expose') and hasattr(oe, 'image'):
                        msg_hist = {'histogram': oe.image,
                                    'sender_name': oe.name,
                                    'sender_id': oeid,
                                    'status': 0}
                        q_out.put(msg_hist)
                    q_out.put({"status": 0, "progress": flowCounter/flowLen})
            handler.bl.forceAlign = False
            q_out.put({"status": 0, "repeat": repeats})
            handler.needUpdate = False
            handler.startEl = None
            time.sleep(0.01)

#            handler.stop = True
            repeats += 1
        else:
            time.sleep(0.1)


class MessageHandler:
    def __init__(self, bl=None):
        self.bl = bl
        self.stop = True
        self.needUpdate = False
        self.autoUpdate = True
        self.startEl = None
        self.exit = False

    def handle_create(self, message):

        objuuid = message.get("uuid")

        object_type = message.get("object_type")
        kwargs = message.get("kwargs", {})

        if object_type == 'beamline':
            self.bl = BeamLine()
            self.bl.deserialize(kwargs)
            self.bl.flowSource = 'Qook'
        elif object_type == 'oe':
            self.bl.init_oe_from_json(kwargs)
        elif object_type == 'mat':
            self.bl.init_material_from_json(kwargs)

        if self.autoUpdate:
            if object_type != 'mat':
                self.needUpdate = True
                self.startEl = objuuid

    def handle_modify(self, message):
        objuuid = message.get("uuid")
        object_type = message.get("object_type")
        kwargs = message.get("kwargs", {})

        if object_type == 'oe':
            eLine = self.bl.oesDict.get(objuuid)

            if eLine is not None:
                element = eLine[0]
                for key, value in kwargs.items():
                    args = key.split('.')
                    arg = args[0]
                    if len(args) > 1:
                        field = args[-1]
                        if field == 'energy':
                            if arg == 'bragg':
                                value = [float(value)]
                            else:
                                value = element.material.get_Bragg_angle(
                                        float(value))
                        else:
                            argIn = getattr(element, f'_{arg}', None)
                            arrayValue = getattr(element, arg) if\
                                argIn is None else argIn

                            if hasattr(arrayValue, 'tolist'):
                                arrayValue = arrayValue.tolist()

                            for fList in compoundArgs.values():
                                if field in fList:
                                    idx = fList.index(field)
                                    break
                            arrayValue[idx] = value
                            value = arrayValue

                    setattr(element, arg, value)
                    if arg.lower().startswith('center'):
                        self.bl.sort_flow()

                if self.autoUpdate:
                    self.needUpdate = True
                    if len(kwargs) == 1 and (kwargs.keys() & renderOnlyArgSet):
                        self.needUpdate = False
                if hasattr(element, 'propagate') and \
                        objuuid in self.bl.flowU:
                    kwargs = list(self.bl.flowU[objuuid].values())[0]
                    modifiedEl = kwargs['beam']
                else:
                    modifiedEl = objuuid

                if self.needUpdate:
                    keys = list(self.bl.flowU.keys())
                    if self.startEl is None:
                        self.startEl = modifiedEl
                    elif keys.index(modifiedEl) < keys.index(self.startEl):
                        self.startEl = modifiedEl
        elif object_type == 'mat':
            # element = self.bl.materialsDict.get(objuuid)
            # We reinstantiate the material object instead of updating. Single-
            # property update not supported yet for materials.
            # object will use the same uuid
            if objuuid in self.bl.materialsDict:
                del self.bl.materialsDict[objuuid]
            self.bl.init_material_from_json(objuuid, kwargs)

            self.startEl = None
            for oeid, oeLine in self.bl.oesDict.items():
                oeObj = oeLine[0]
                for prop in ["_material", "_material2"]:
                    try:
                        matProp = getattr(oeObj, prop)
                        if matProp == objuuid:
                            self.startEl = oeid
                            break
                    except AttributeError:
                        pass

            if self.autoUpdate and self.startEl is not None:
                self.needUpdate = True

    def handle_flow(self, message):
        oeuuid = message.get('uuid')
        kwargs = message.get('kwargs')
#        print("handle_flow", message)
        self.bl.update_flow_from_json(oeuuid, kwargs)
        self.bl.sort_flow()
        if self.autoUpdate:
            self.needUpdate = True
            self.startEl = oeuuid

    def handle_delete(self, message):
        objuuid = message.get("uuid")
        object_type = message.get("object_type")
        if object_type == "oe":
            self.bl.delete_oe_by_id(objuuid)
            if self.autoUpdate:
                self.needUpdate = True
                self.startEl = objuuid
        elif object_type == "mat":
            self.startEl = None
            for oeid, oeLine in self.bl.oesDict.items():
                oeObj = oeLine[0]
                for prop in ["_material", "_material2"]:
                    try:
                        matProp = getattr(oeObj, prop)
                        if matProp == objuuid:
                            self.startEl = oeid
                            break
                    except AttributeError:
                        pass
            self.bl.delete_mat_by_id(objuuid)

            if self.autoUpdate and self.startEl is not None:
                self.needUpdate = True

    def handle_start(self, message):
        print("Starting processing loop.")
        self.stop = False
        self.needUpdate = True

    def handle_exit(self, message):
        print("Exiting.")
        self.exit = True

    def handle_run_once(self, message):
        print("Starting processing loop.")
        self.needUpdate = True
        self.startEl = None

    def handle_auto_update(self, message):
        # print("Starting processing loop.")
        kwargs = message.get('kwargs')
        if kwargs is not None:
            auto_update = kwargs.get('value')

        if bool(auto_update):
            self.needUpdate = True
            self.startEl = None

        self.autoUpdate = bool(auto_update)

    def handle_stop(self, message):
        print("Stopping processing loop.")
        self.stop = True

    def process_message(self, message):
        # Build a dispatch dictionary mapping commands to methods.
        command_handlers = {
            "create": self.handle_create,
            "modify": self.handle_modify,
            "delete": self.handle_delete,
            "start": self.handle_start,
            "stop": self.handle_stop,
            "exit": self.handle_exit,
            "flow": self.handle_flow,
            "run_once": self.handle_run_once,
            "auto_update": self.handle_auto_update,
        }

        command = message.get("command")
        handler = command_handlers.get(command)
        if handler:
            # print(message)
            handler(message)
        else:
            print(f"Unknown command: {command}")
