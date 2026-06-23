/**
 * Tests for the Anthropic Claude Compliance API integration mapper.
 */

import { ClaudeComplianceMapper, classify } from "./claude-compliance";

describe("classify", () => {
  it("classifies authentication activity", () => {
    expect(classify("sso_login_initiated")).toEqual({
      categoryUid: 3,
      classUid: 3002,
      activityId: 1,
      category: "authentication",
    });
    expect(classify("user_sign_out")).toEqual({
      categoryUid: 3,
      classUid: 3002,
      activityId: 2,
      category: "authentication",
    });
  });

  it("classifies account change activity", () => {
    const r = classify("user_added");
    expect([r.categoryUid, r.classUid]).toEqual([3, 3001]);
    expect(classify("user_removed")).toEqual({
      categoryUid: 3,
      classUid: 3001,
      activityId: 4,
      category: "account_change",
    });
  });

  it("classifies access management activity", () => {
    expect(classify("role_permission_granted")).toEqual({
      categoryUid: 3,
      classUid: 3005,
      activityId: 1,
      category: "access_management",
    });
    const r = classify("group_member_added");
    expect([r.categoryUid, r.classUid]).toEqual([3, 3005]);
  });

  it("classifies content activity", () => {
    expect(classify("claude_chat_created")).toEqual({
      categoryUid: 6,
      classUid: 6001,
      activityId: 1,
      category: "content",
    });
    expect(classify("claude_file_uploaded")).toEqual({
      categoryUid: 6,
      classUid: 6001,
      activityId: 1,
      category: "content",
    });
    expect(classify("chat_message_viewed")).toEqual({
      categoryUid: 6,
      classUid: 6001,
      activityId: 2,
      category: "content",
    });
  });

  it("falls back to API Activity / Other for unknown types", () => {
    // New/unknown types fall back to API Activity / Other, never dropped.
    expect(classify("some_brand_new_future_event")).toEqual({
      categoryUid: 6,
      classUid: 6003,
      activityId: 99,
      category: "other",
    });
  });
});

describe("ClaudeComplianceMapper", () => {
  let mapper: ClaudeComplianceMapper;

  beforeEach(() => {
    mapper = new ClaudeComplianceMapper();
  });

  it("maps chat_created to web resources activity", () => {
    const event = mapper.mapActivity({
      id: "activity_1",
      created_at: "2026-04-10T08:09:10Z",
      organization_id: "org_1",
      actor: {
        type: "user_actor",
        email_address: "u@example.com",
        user_id: "user_1",
        ip_address: "192.0.2.34",
        user_agent: "Mozilla/5.0",
      },
      type: "claude_chat_created",
      claude_chat_id: "chat_1",
      claude_project_id: "proj_1",
    });
    expect(event.category_uid).toBe(6);
    expect(event.class_uid).toBe(6001);
    expect(event.type_uid).toBe(600101);
    expect(event.time).toBe("2026-04-10T08:09:10Z");
    expect(event.actor?.user?.uid).toBe("user_1");
    expect(event.actor?.user?.email_addr).toBe("u@example.com");
    expect(event.device?.ip).toBe("192.0.2.34");
    const names = Object.fromEntries(event.enrichments.map((e) => [e.name, e.value]));
    expect(names["claude.compliance.chat.id"]).toBe("chat_1");
    expect(names["claude.compliance.activity.type"]).toBe("claude_chat_created");
    const obs = new Set(event.observables.map((o) => o.value));
    expect(obs.has("192.0.2.34")).toBe(true);
    expect(obs.has("u@example.com")).toBe(true);
  });

  it("maps login to authentication", () => {
    const event = mapper.mapActivity({
      id: "a2",
      created_at: "2026-04-10T09:00:00Z",
      actor: {
        type: "unauthenticated_user_actor",
        unauthenticated_email_address: "x@example.com",
        ip_address: "10.0.0.1",
      },
      type: "sso_login_initiated",
    });
    expect([event.category_uid, event.class_uid, event.activity_id]).toEqual([3, 3002, 1]);
    expect(event.metadata.product.vendor_name).toBe("Anthropic");
  });

  it("maps scim user_added to account change with directory_id uid", () => {
    const event = mapper.mapActivity({
      id: "a3",
      actor: { type: "scim_directory_sync_actor", directory_id: "dir_1" },
      type: "user_added",
    });
    expect([event.category_uid, event.class_uid]).toEqual([3, 3001]);
    expect(event.actor?.user?.uid).toBe("dir_1");
    // No created_at -> falls back to a generated timestamp (non-empty).
    expect(event.time).toBeTruthy();
  });

  it("detects failure status", () => {
    const event = mapper.mapActivity({
      id: "a4",
      actor: { type: "api_actor" },
      type: "sso_login_failed",
    });
    expect(event.status_id).toBe(2); // Failure
  });

  it("maps a batch of activities", () => {
    const events = mapper.mapActivities([
      { id: "x1", actor: { type: "api_actor" }, type: "claude_file_uploaded" },
      { id: "x2", actor: { type: "api_actor" }, type: "compliance_export_created" },
    ]);
    expect(events.map((e) => e.class_uid)).toEqual([6001, 6003]);
  });
});
