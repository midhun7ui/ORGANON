
import pytest

from core.security.base import UserContext
from core.security.resource_acl import (
    InMemoryResourceACLProvider,
    ResourceACLRule,
    ResourceContext,
)

import pytest

from core.security.resource_acl import ResourceContext

@pytest.fixture
def user():
    return UserContext(
        user_id="user-1",
        tenant_id="tenant-1",
        roles={"editor"},
    )


@pytest.fixture
def resource():
    return ResourceContext(
        resource_id="doc-1",
        resource_type="document",
        tenant_id="tenant-1",
    )


@pytest.fixture
def acl_provider():
    return InMemoryResourceACLProvider(
        rules={
            ("tenant-1", "document", "doc-1"): (
                ResourceACLRule(
                    principal_type="user",
                    principal_id="user-1",
                    actions=frozenset({"read"}),
                ),
                ResourceACLRule(
                    principal_type="role",
                    principal_id="editor",
                    actions=frozenset({"update"}),
                ),
            )
        }
    )


@pytest.mark.asyncio
async def test_user_can_access_resource(user, resource, acl_provider):
    assert await acl_provider.authorize(user, resource, "read") is True


@pytest.mark.asyncio
async def test_role_can_access_resource(user, resource, acl_provider):
    assert await acl_provider.authorize(user, resource, "update") is True


@pytest.mark.asyncio
async def test_ungranted_action_is_denied(user, resource, acl_provider):
    assert await acl_provider.authorize(user, resource, "delete") is False


@pytest.mark.asyncio
async def test_unknown_resource_is_denied(user, acl_provider):
    unknown_resource = ResourceContext(
        resource_id="unknown-doc",
        resource_type="document",
        tenant_id="tenant-1",
    )

    assert await acl_provider.authorize(
        user, unknown_resource, "read"
    ) is False


@pytest.mark.asyncio
async def test_cross_tenant_access_is_denied(user, acl_provider):
    other_tenant_resource = ResourceContext(
        resource_id="doc-1",
        resource_type="document",
        tenant_id="tenant-2",
    )

    assert await acl_provider.authorize(
        user, other_tenant_resource, "read"
    ) is False


@pytest.mark.asyncio
async def test_wrong_user_is_denied(resource, acl_provider):
    another_user = UserContext(
        user_id="user-2",
        tenant_id="tenant-1",
    )

    assert await acl_provider.authorize(
        another_user, resource, "read"
    ) is False


@pytest.mark.asyncio
async def test_invalid_inputs_are_denied(acl_provider, user, resource):
    assert await acl_provider.authorize(
        None, resource, "read"
    ) is False

    assert await acl_provider.authorize(
        user, None, "read"
    ) is False

    assert await acl_provider.authorize(
        user, resource, ""
    ) is False

@pytest.mark.parametrize(
    "field_values",
    [
        {"resource_id": ""},
        {"resource_id": "   "},
        {"resource_type": ""},
        {"resource_type": "   "},
        {"tenant_id": ""},
        {"tenant_id": "   "},
        {"owner_id": ""},
        {"owner_id": "   "},
    ],
)
def test_resource_context_rejects_empty_fields(field_values):
    values = {
        "resource_id": "doc-1",
        "resource_type": "document",
        "tenant_id": "tenant-1",
    }
    values.update(field_values)

    with pytest.raises(ValueError):
        ResourceContext(**values)


@pytest.mark.parametrize(
    "field_values",
    [
        {"resource_id": None},
        {"resource_type": 123},
        {"tenant_id": None},
        {"owner_id": 123},
    ],
)
def test_resource_context_rejects_invalid_field_types(field_values):
    values = {
        "resource_id": "doc-1",
        "resource_type": "document",
        "tenant_id": "tenant-1",
    }
    values.update(field_values)

    with pytest.raises(ValueError):
        ResourceContext(**values)


def test_resource_context_allows_optional_owner():
    resource = ResourceContext(
        resource_id="doc-1",
        resource_type="document",
        tenant_id="tenant-1",
    )

    assert resource.owner_id is None


@pytest.mark.parametrize(
    "principal_type",
    ["", "   ", "group", None, 123],
)
def test_acl_rule_rejects_invalid_principal_type(principal_type):
    with pytest.raises(ValueError):
        ResourceACLRule(
            principal_type=principal_type,
            principal_id="user-1",
            actions={"read"},
        )


@pytest.mark.parametrize(
    "principal_id",
    ["", "   ", None, 123],
)
def test_acl_rule_rejects_invalid_principal_id(principal_id):
    with pytest.raises(ValueError):
        ResourceACLRule(
            principal_type="user",
            principal_id=principal_id,
            actions={"read"},
        )


@pytest.mark.parametrize(
    "actions",
    [
        set(),
        frozenset(),
        {"read", ""},
        {"read", "   "},
        {"read", 123},
        ["read"],
    ],
)
def test_acl_rule_rejects_invalid_actions(actions):
    with pytest.raises(ValueError):
        ResourceACLRule(
            principal_type="user",
            principal_id="user-1",
            actions=actions,
        )


def test_acl_rule_converts_actions_to_frozenset():
    rule = ResourceACLRule(
        principal_type="user",
        principal_id="user-1",
        actions={"read", "update"},
    )

    assert isinstance(rule.actions, frozenset)
    assert rule.actions == frozenset({"read", "update"})


def test_acl_provider_rejects_invalid_rules_mapping():
    with pytest.raises(TypeError):
        InMemoryResourceACLProvider(rules=[])


@pytest.mark.parametrize(
    "resource_key",
    [
        ("tenant-1", "document"),
        ("tenant-1", "document", ""),
        ("tenant-1", "document", None),
        ("tenant-1", 123, "doc-1"),
    ],
)
def test_acl_provider_rejects_invalid_resource_keys(resource_key):
    with pytest.raises(ValueError):
        InMemoryResourceACLProvider(
            rules={resource_key: ()}
        )


def test_acl_provider_rejects_invalid_rule_collection():
    with pytest.raises(TypeError):
        InMemoryResourceACLProvider(
            rules={
                ("tenant-1", "document", "doc-1"): "invalid"
            }
        )


def test_acl_provider_rejects_invalid_rule_entries():
    with pytest.raises(TypeError):
        InMemoryResourceACLProvider(
            rules={
                ("tenant-1", "document", "doc-1"): ("invalid-rule",)
            }
        )


def test_acl_provider_stores_rules_immutably():
    original_rules = {
        ("tenant-1", "document", "doc-1"): (
            ResourceACLRule(
                principal_type="user",
                principal_id="user-1",
                actions={"read"},
            ),
        )
    }

    provider = InMemoryResourceACLProvider(original_rules)

    original_rules.clear()

    assert len(provider.rules) == 1

    with pytest.raises(TypeError):
        provider.rules[("tenant-2", "document", "doc-2")] = ()
