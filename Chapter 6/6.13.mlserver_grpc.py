import json
import grpc
from typing import Optional
from mlserver.codecs.string import StringRequestCodec
import mlserver.grpc.converters as converters
import mlserver.grpc.dataplane_pb2_grpc as dataplane
import mlserver.types as types

def create_request(inputs: dict[str, str]) -> types.InferenceRequest:
    input_bytes = json.dump(inputs).encode("UTF-8")

    return types.InferenceRequest(
        input=[
            types.RequestInput(
                name="request",
                shape=[len(input_bytes)],
                datatype="BYTES",
                data=[input_bytes],
                parameters=types.Parameters(content_type="str"),
            )
        ]
    )

def serilize_request(
        request:types.InferenceRequest,
        model_name: str,
        model_version: Optional[str] = None,
) -> bytes:
    return converters.ModelInferRequestConverter.from_types(
        request, model_name=model_name,model_version=model_version
    )

def connect_grpc(host: str) -> dataplane.GRPCInferenceServiceStub:
    grpc_channel = grpc.insecure_channel(host)
    return dataplane.GRPCInferenceServiceStub(grpc_channel)


def make_grpc_request(
        grpc_stub:dataplane.GRPCInferenceServiceStub, serialized_request: bytes
) -> dataplane.ModelInferResponse:

    return grpc_stub.ModelInfer(serialized_request)

def deserialize_response(
        response: dataplane.ModelInferResponse,
) -> str:
    deserialize_response = converters.ModelInferResponseConverter.to_types(
        response
    )

    return StringRequestCodec.decode_response(deserialize_response)

if __name__=="__main__":

    model_name = "grpc_model"
    inputs = {"message": "I'm using gRPC!"}

    request = create_request(inputs)
    serialized_request = serilize_request(request, model_name)
    grpc_stub = connect_grpc("localhost:8081")
    response = make_grpc_request(grpc_stub, serialized_request)
    print(response)

    json_text = deserialize_response(response)
    output = json.load(json_text[0])
    print(output)