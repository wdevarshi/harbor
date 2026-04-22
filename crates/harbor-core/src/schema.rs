use crate::{JsonValue, ToolSpec};
use serde_json::{json, Map, Value};

pub trait TypedSchema {
    fn schema() -> JsonValue;
}

#[derive(Debug, Clone)]
pub struct Schema {
    value: JsonValue,
}

impl Schema {
    pub fn new(value: JsonValue) -> Self {
        Self { value }
    }

    pub fn string() -> Self {
        Self::new(json!({ "type": "string" }))
    }

    pub fn integer() -> Self {
        Self::new(json!({ "type": "integer" }))
    }

    pub fn number() -> Self {
        Self::new(json!({ "type": "number" }))
    }

    pub fn boolean() -> Self {
        Self::new(json!({ "type": "boolean" }))
    }

    pub fn null() -> Self {
        Self::new(json!({ "type": "null" }))
    }

    pub fn array(item: Schema) -> Self {
        Self::new(json!({
            "type": "array",
            "items": item.into_json(),
        }))
    }

    pub fn enumeration<I, V>(values: I) -> Self
    where
        I: IntoIterator<Item = V>,
        V: Into<Value>,
    {
        Self::new(json!({
            "enum": values.into_iter().map(Into::into).collect::<Vec<_>>(),
        }))
    }

    pub fn object() -> ObjectSchemaBuilder {
        ObjectSchemaBuilder::default()
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        if let Some(object) = self.value.as_object_mut() {
            object.insert("description".into(), Value::String(description.into()));
        }
        self
    }

    pub fn nullable(self) -> Self {
        let mut value = self.value;
        let type_value = value.get("type").cloned();
        if let Some(Value::String(schema_type)) = type_value {
            if let Some(object) = value.as_object_mut() {
                object.insert(
                    "type".into(),
                    Value::Array(vec![Value::String(schema_type), Value::String("null".into())]),
                );
            }
        }
        Self::new(value)
    }

    pub fn into_json(self) -> JsonValue {
        self.value
    }
}

#[derive(Debug, Clone, Default)]
pub struct ObjectSchemaBuilder {
    properties: Map<String, Value>,
    required: Vec<String>,
    description: Option<String>,
    additional_properties: Option<bool>,
}

impl ObjectSchemaBuilder {
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    pub fn property(mut self, name: impl Into<String>, schema: Schema) -> Self {
        self.properties.insert(name.into(), schema.into_json());
        self
    }

    pub fn required_property(mut self, name: impl Into<String>, schema: Schema) -> Self {
        let name = name.into();
        self.required.push(name.clone());
        self.properties.insert(name, schema.into_json());
        self
    }

    pub fn additional_properties(mut self, allowed: bool) -> Self {
        self.additional_properties = Some(allowed);
        self
    }

    pub fn build(self) -> Schema {
        let mut object = Map::new();
        object.insert("type".into(), Value::String("object".into()));
        object.insert("properties".into(), Value::Object(self.properties));

        if !self.required.is_empty() {
            object.insert(
                "required".into(),
                Value::Array(self.required.into_iter().map(Value::String).collect()),
            );
        }

        if let Some(description) = self.description {
            object.insert("description".into(), Value::String(description));
        }

        if let Some(additional_properties) = self.additional_properties {
            object.insert(
                "additionalProperties".into(),
                Value::Bool(additional_properties),
            );
        }

        Schema::new(Value::Object(object))
    }
}

pub fn typed_tool_spec<T>(name: impl Into<String>, description: impl Into<String>) -> ToolSpec
where
    T: TypedSchema,
{
    ToolSpec::new(name, description, T::schema())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct CreateNoteSchema;

    impl TypedSchema for CreateNoteSchema {
        fn schema() -> JsonValue {
            Schema::object()
                .required_property("title", Schema::string().with_description("Note title"))
                .property("body", Schema::string())
                .required_property("published", Schema::boolean())
                .additional_properties(false)
                .build()
                .into_json()
        }
    }

    #[test]
    fn object_schema_builder_generates_expected_shape() {
        let schema = Schema::object()
            .description("Create a note")
            .required_property("title", Schema::string())
            .property("tags", Schema::array(Schema::string()))
            .additional_properties(false)
            .build()
            .into_json();

        assert_eq!(schema["type"], "object");
        assert_eq!(schema["description"], "Create a note");
        assert_eq!(schema["required"][0], "title");
        assert_eq!(schema["properties"]["tags"]["type"], "array");
        assert_eq!(schema["properties"]["tags"]["items"]["type"], "string");
        assert_eq!(schema["additionalProperties"], false);
    }

    #[test]
    fn typed_tool_spec_uses_typed_schema() {
        let spec = typed_tool_spec::<CreateNoteSchema>("create_note", "Create note");
        assert_eq!(spec.name, "create_note");
        assert_eq!(spec.description, "Create note");
        assert_eq!(spec.input_schema["type"], "object");
        assert_eq!(spec.input_schema["properties"]["title"]["description"], "Note title");
        assert_eq!(spec.input_schema["required"][0], "title");
        assert_eq!(spec.input_schema["required"][1], "published");
    }

    #[test]
    fn enum_and_nullable_helpers_work() {
        let schema = Schema::string().with_description("Status").nullable().into_json();

        assert_eq!(schema["description"], "Status");
        assert_eq!(schema["type"][0], "string");
        assert_eq!(schema["type"][1], "null");
    }
}
